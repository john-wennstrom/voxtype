# Voxtype phonetic retrieval for domain terms — implementation plan

Oct 9, 2026 · @John

## Goals and scope

Voxtype gets a phonetic retrieval layer that maps uncertain ASR output to terms in a large personal lexicon (programming and IT jargon, English only), in under 1 ms per utterance, in both push-to-talk and streaming mode.

**In scope**

- An offline G2P script that turns the term list into phoneme variants.
- A Rust index builder and a memory-mapped phonetic index (phonetic keys plus a phoneme q-gram index).
- Per-user and per-project lexicons, merged at query time.
- A query pipeline: confidence gating, 1–3 word windows, candidate retrieval, weighted phoneme rerank.
- An LLM selection layer that picks among retrieved candidates with TypeSafe Jev's Choice primitive, with a local GGUF fallback.
- Two-pass decoding with retrieved boost lists for push-to-talk.
- Nemotron streaming integration: rolling boost list, hold-back before typing, end-of-utterance pass.

**Out of scope:** the Nemotron model integration itself (already done) and multilingual support.

**What the plan builds on**:

1. Voxtype is a Rust application and the Nemotron backend runs in-process: a GGUF model loaded through the transcribe.cpp 0.3.1 Rust binding.
2. The backend already returns per-word text, start/end time and confidence.
3. transcribe.cpp's public C header (checked on its main branch, not on 0.3.1) has no word boosting and no encoder-only or decoder-only calls. Boosting and encoder reuse therefore need a small patch to transcribe.cpp (section 8). The text-level and LLM layers work with the API as it is.
4. Recognised text is typed into the focused window, so anything typed is hard to take back. This drives the streaming commit rules.

**Performance targets**

| Metric | Target |
| --- | --- |
| Lexicon size | about 30k terms (user + project), about 100k variants; benchmarked to 1M for headroom |
| Index open (mmap) | < 1 ms, no parse step |
| Query per window, 100k variants | p50 < 20 µs, p99 < 100 µs |
| Post-processing per utterance (text-level fix) | < 1 ms |
| LLM selection, when triggered | p95 < 300 ms push-to-talk (network-bound, measure first); end of utterance only when streaming |
| Second pass (push-to-talk) | decoder-only on uncertain spans, < 15% of first-pass time |
| Streaming added latency | ≤ hold-back of 3 words or end of utterance |
| Resident memory, index | < 16 MB at 100k variants |
| Allocations in the query hot path | zero after warm-up |

## Architecture

Two halves: an offline pipeline that turns your term list into a memory-mapped index, and a runtime path where a lexicon worker sits between the Nemotron decoder and typing, correcting text and feeding boosts back into the decoder.

&#91;embedded content: voxtype lexicon architecture · offline build and runtime path\]

The index is only read at runtime; rebuilding it never pauses dictation. The feedback loop from the lexicon worker through the rolling boost set is what lets the audio, not just the text, decide between candidates.

**Code layout**

```
tools/lexicon_g2p.py          offline G2P (Python, not shipped in the hot path)
crates/voxtype-lexicon/
  src/format.rs               .vxl sections, header, validation on open
  src/build.rs                TSV → ids by length → keys, q-grams → CSR
  src/index.rs                PhoneticIndex: open, g2p_word, search
  src/alphabet.rs             fine/coarse phoneme ids, substitution table
  src/query.rs                gate, windows, rerank, interval selection
  src/context_graph.rs        token trie with bonuses (if the decoder lacks one)
  src/stream.rs               word states, hold-back, rolling boost set
  benches/                    criterion benches at 10k / 100k / 1M
src/ (voxtype)                hooks in the Nemotron backend and typing pipeline
```

## Lexicon format and G2P script

G2P runs offline in Python (`tools/lexicon_g2p.py`) and never in the hot path. It writes one ARPAbet phoneme string per pronunciation variant, which the Rust builder compiles into the index.

**Input: `terms.tsv`** (one term per line, `#` comments, extra columns optional)

```
# term        spoken forms (| separated)          weight
kubectl       cube control | cube c t l | kube cuttle   1.0
nginx         engine x                                 1.0
PostgreSQL    postgres q l | postgres                  1.0
useEffect                                              0.8
JSON
```

**Phoneme alphabet:** the 39 ARPAbet phonemes with stress removed (`AH0` → `AH`). Each maps to a `u8` id at build time. Stress is dropped because ASR output rarely carries reliable stress, and keeping it splits matches.

**Variant generation, in priority order** (deduplicated per term, capped at 8 by default):

1. **User spoken forms.** Each spoken form is converted word by word. These are the most valuable variants, so they are never dropped by the cap.
2. **CMUdict.** Every listed pronunciation of the whole term.
3. **Identifier splitting.** camelCase, snake\_case, kebab-case, dots and digit runs are split, and each part is looked up: `useEffect` → use + effect, `.env` → dot + env.
4. **Acronyms.** All-caps tokens of 2–5 letters, or tokens with no vowels, get a spelled-out variant from letter names (`SQL` → S Q L), and also a whole-word variant (`JSON` → jason).
5. **Symbols.** `.` dot, `_` underscore, `-` dash, `/` slash, `#` sharp, `+` plus, `@` at; digits become number words. Each identifier also gets a variant with symbols silent, since \`CI/CD\` is usually said "C I C D".
6. **Neural fallback** for parts CMUdict lacks: `g2p_en` by default, `espeak-ng` (IPA mapped to ARPAbet) as an alternative or extra variant with `--backend all`.

**Outputs**

- `lexicon.phon.tsv`: `term_id · canonical · source · phonemes · weight`, one row per variant. `source` (`user`, `cmudict`, `split`, `acronym`, `g2p_en`, `espeak`) lets the scorer trust user variants more.
- `querydict.phon.tsv` (`--export-query-dict`): normalised CMUdict, word → phonemes. The builder embeds it in the index so the runtime can convert ASR words to phonemes with one FST lookup and no Python.

**Performance of the script:** results are cached per word part, and neural G2P runs in a process pool. 100k terms should take minutes, not hours, and it only reruns when `terms.tsv` changes.

## Rust phonetic index

The index is one immutable, memory-mapped file (`.vxl`) with three lookup layers: an exact phonetic-key hash, a direct-indexed phoneme q-gram table for fuzzy candidates, and a weighted phoneme edit distance to verify them. Nothing is parsed on load; every section is cast in place.

**Crate:** `voxtype-lexicon` (workspace member), with `build` and `query` modules and a `voxtype lexicon build` subcommand. Dependencies: `memmap2`, `bytemuck`, `fst`, `xxhash-rust`, `rayon` (build only), `smallvec`, `arc-swap`.

### Two alphabets

- **Fine:** 39 ARPAbet phonemes as `u8` ids. Used for verification.
- **Coarse:** 15 classes used for keys and q-grams, so typical ASR confusions land in the same bucket: all vowels → `V`, `P/B`, `T/D`, `K/G`, `F/V`, `TH/DH`, `S/Z`, `SH/ZH`, `CH/JH`, `M/N/NG`, `L`, `R/ER`, `W`, `Y`, `HH`. Each fits in 4 bits.

### Layer 1: phonetic key (exact, O(1))

The key is the consonant skeleton: coarse classes, vowels dropped, repeats collapsed, word boundaries removed. "cube control" and the `kubectl` variant both become `K Y B K N T R L`. Packed at 4 bits per class into a `u64` (up to 16 classes; longer keys are hashed with xxh3). Stored as an open-addressing table (power-of-two size, linear probing, load ≤ 0.7) pointing into a CSR array of variant ids. One lookup answers the common case of a near-perfect phonetic match.

### Layer 2: q-gram candidates (fuzzy)

- Coarse trigrams with vowels kept as `V` and boundary padding: 16³ = 4,096 possible grams, so the table is a **direct-indexed `[u32; 4097]` offset array**, with no hashing. Postings are `u32` variant ids.
- **Variant ids are assigned in order of phoneme length.** A `len_start[64]` table then turns the length filter (|len − L| ≤ k) into one id range, and each sorted posting list is clipped to it with two binary searches (`partition_point`). Most postings are never touched.
- **Count filter:** with k edits, a match must share at least |grams| − 3k grams. Counting uses a per-thread scratch of `u16` counts plus `u32` generation stamps (no clearing between queries) and a touched list.
- **Stop grams:** grams with document frequency above about 5% are skipped at query time and the count threshold lowered to match. This caps the cost of very common grams like `V N V`.

### Layer 3: verification

Weighted Levenshtein over fine phoneme ids, with a 40×40 `u8` substitution table: same coarse class 1, vowel ↔ vowel 2, other 4, insert/delete 3. The table starts hand-set and is later learned from real confusions (section 11). The DP runs on stack arrays (phoneme strings are under 64), banded to k, and exits early once a row's minimum exceeds the current k-th best. Optional: Myers bit-parallel unweighted distance on a `u64` as a cheaper prefilter, if benchmarks show verification dominates.

### File layout (`.vxl`, little-endian, sections 64-byte aligned)

| Section | Contents |
| --- | --- |
| Header | magic, version, counts, alphabet hash, tokenizer hash, xxh3 per section |
| Phoneme arena | `u8` fine ids, all variants back to back |
| Variants | `#[repr(C)]` {arena offset `u32`, term id `u32`, len `u8`, source `u8`, pad} = 12 B, sorted by length |
| Length starts | `[u32; 65]` |
| Q-gram offsets / postings | `[u32; 4097]` + `u32` ids |
| Key table / postings | {key `u64`, offset `u32`, len `u32`} slots + `u32` ids |
| Terms | canonical string offset/len, weight (`u16` quantised), boost-token offset/len |
| Strings / tokens | UTF-8 canonical forms; Nemotron SentencePiece ids (`u16`) per term, pre-tokenised |
| Query dictionary | `fst::Map` word → packed (offset, len) into its own phoneme arena |

Pre-tokenising terms at build time means the runtime never runs the tokenizer for boosting. The tokenizer hash in the header makes a stale index fail to load instead of boosting wrong tokens.

**Size at your scale:** 30k terms with about 3 variants each is roughly 100k variants: arena about 1 MB, variants 1.2 MB, postings about 4 MB, keys under 1 MB, plus the query dictionary (about 3 MB). The whole index stays in cache-friendly territory. Benchmarks still run to 1M variants so a growing lexicon never surprises you.

**Is the q-gram layer still needed at 30k?** Probably yes. A brute-force weighted DP over 100k variants costs several milliseconds per window, against tens of microseconds with q-gram filtering. The `qgram_search` vs brute-force bench (section 10) confirms it before the layer is built.

### Build

Parse TSV → assign ids by length → extract grams and keys in parallel with `rayon` → radix-sort (gram, id) pairs → write CSR sections → write to a temp file, `fsync`, rename. The running app watches the file and swaps indexes atomically with `ArcSwap`, so in-flight queries finish on the old one.

### API

```rust
pub struct PhoneticIndex { /* Mmap + section slices */ }

impl PhoneticIndex {
    pub fn open(path: &Path) -> Result<Self>;
    /// Query-side G2P: one FST lookup per ASR word.
    pub fn g2p_word(&self, word: &str, out: &mut SmallVec<[u8; 32]>) -> bool;
    pub fn search(&self, phones: &[u8], opts: &SearchOpts,
                  scratch: &mut Scratch, out: &mut Vec<Candidate>);
    pub fn term(&self, id: u32) -> TermRef<'_>; // canonical, weight, boost tokens
}

pub struct Candidate { pub term: u32, pub variant: u32, pub cost: u16, pub sim: f32 }
```

## Query pipeline

Only uncertain spans are searched, each as 1–3 word windows, and the best non-overlapping set of replacements is chosen in one pass. A typical utterance has zero to two uncertain spans, so most utterances cost microseconds.

**Input** per word from Nemotron: text, start/end time, confidence. Output: corrected words plus a candidate set per span (used by the second pass, streaming boosting and an optional LLM stage).

1. **Confidence gate.** A word is uncertain if its confidence is below τ (calibrated per model, start at 0.6), or if it is missing from the query dictionary (the ASR produced an odd token). Very common short words ("the", "a") need a stricter τ, because they rarely hide jargon but often look uncertain.
2. **Windows.** For each uncertain word, form every window of 1–3 consecutive words that contains it. Overlapping windows from neighbouring uncertain words are deduplicated. "run cube control apply" with "cube" and "control" uncertain gives `cube`, `control`, `cube control`, `run cube`, `control apply`, `run cube control`, `cube control apply`.
3. **Query G2P.** Each word → phonemes via the embedded FST (one lookup, no allocation). Words are concatenated without boundaries, because jargon rarely splits where English words do. Out-of-dictionary words fall back to a small letter-to-sound rule set in Rust; it is crude, but the q-gram layer tolerates it.
4. **Retrieve.** Exact key lookup first. If it returns fewer than k candidates (k = 8), run the q-gram search with an edit budget proportional to length (about 1 edit per 4 phonemes).
5. **Rerank.** `score = sim × source_trust × term_weight`, where `sim = 1 − cost / max_cost(len)`. User-supplied spoken variants get the highest source trust.
6. **Select.** Weighted interval scheduling over all window candidates picks the non-overlapping set with the best total gain. Gain = score − the original words' confidence, so a confident original is hard to override. A replacement needs `sim ≥ θ` (start at 0.82). Below θ, candidates are still returned for the second pass.
7. **Emit.** Replace the words with the canonical spelling and keep the original text, scores and time span alongside, so a UI or log can show what changed.

**Hot-path rules**

- One `Scratch` per thread holds every buffer (phoneme buffers, counters, stamps, candidate heap). No allocation after warm-up.
- An LRU cache (about 4k entries) maps window phonemes → top-k results. Dictation repeats words, so hit rates should be high.
- Results are pure functions of (index, window phonemes, options), so caching and testing are simple.

**Context guard:** a replacement that turns a common word into jargon ("get" → `git`) needs a neighbouring cue to fire at text level: another term in the same utterance, or a recent hit in the session. Otherwise it is only passed on as a candidate for the second pass, where the audio decides.

## Per-user and per-project lexicons

Each lexicon is built into its own index and both are queried, with results merged by term; a project entry wins over a user entry for the same term. Separate indexes mean switching projects never rebuilds your user index, and a project index (a few thousand terms) rebuilds in milliseconds.

**Files**

| Lexicon | Source (edit this) | Index (generated) |
| --- | --- | --- |
| User | `~/.config/voxtype/lexicon/terms.tsv` | `~/.cache/voxtype/lexicon/user.vxl` |
| Project | `<project>/.voxtype/terms.tsv` | `~/.cache/voxtype/lexicon/<hash of project path>.vxl` |

The project `terms.tsv` can be committed so a team shares it; the index stays in your cache, out of the repo.

**Merge rules** (key = canonical term, case-folded)

- Project entry present → its spelling, casing and weight win; spoken forms are the union of both.
- A project line `!term` suppresses a user term in that project (say, a term that collides with ordinary words there).
- Candidates from both indexes are merged by key, keeping the best similarity per term. At equal similarity the project candidate ranks first.
- The static boost list takes the project's top terms first, then fills from the user lexicon.

Suppressions and overrides are applied as a small hash set and map at load time, not baked into either index, so the user index never depends on which project is active.

**Runtime shape**

```rust
pub struct LexiconSet {
    user: Arc<PhoneticIndex>,
    project: Option<Arc<PhoneticIndex>>,
    overrides: FxHashMap<FoldedTerm, Override>, // spelling, weight, suppressed
}
// swapped atomically on edit or project switch
static ACTIVE: ArcSwap<LexiconSet>;
```

Querying two indexes doubles retrieval cost, which is still tens of microseconds per window.

**Which project is active**

1. Explicit: `voxtype lexicon use <path>` or a config entry; always wins.
2. Automatic (opt-in): take the focused window's process id from the compositor, read its working directory (for terminals, the shell child's `/proc/<pid>/cwd`), and walk up to the nearest `.voxtype/terms.tsv`. Cached per window, refreshed on focus change. How to get the focused PID differs by compositor, so this ships after the explicit mode.
3. None found: user lexicon only.

## LLM selection layer

TypeSafe's Jev model decides only the spans the lexicon can't settle, through its Choice primitive: one Choice question per span, options = "keep original" plus the retrieved terms. Jev returns a probability for every option and a confidence, so the answer is a typed value by design and voxtype never parses free text ([docs](https://docs.typesafe.ai/introduction)).

**When it runs** (per span, after retrieval)

- The top candidate's similarity is between θ\_low (about 0.65) and θ, or the top two are within a small margin of each other.
- The context guard blocked a common-word → jargon replacement ("get" vs `git`).
- Clear hits and spans with no candidates skip it. Most utterances never reach the LLM.

**Request:** one `POST https://api.typesafe.ai/v1/systemone` per utterance, all ambiguous spans as separate questions in the same call ([API reference](https://docs.typesafe.ai/api.md)).

```json
{
  "model": "jev-latest",
  "state": {
    "previous": "we deploy everything to the staging cluster",
    "text": "then run [1: cube control] apply on the [2: helm] chart"
  },
  "questions": {
    "span_1": {
      "type": "choice",
      "instructions": "Which word did the speaker most likely say at [1]?",
      "criteria": {
        "keep: cube control": "the transcription is already right",
        "kubectl": "Kubernetes command-line tool",
        "kubeconfig": "Kubernetes client configuration file",
        "kustomize": "Kubernetes configuration customisation tool"
      }
    },
    "span_2": { "type": "choice", "instructions": "…at [2]?", "criteria": { "keep: helm": null, "Helm": "Kubernetes package manager", "Helmfile": "declarative Helm releases" } }
  }
}
```

Option descriptions come from an optional fourth column in `terms.tsv`. They help most when two terms sound alike but mean different things. A Choice question takes up to 255 options, far more than the top 5 per span sent here.

**Response, typed in Rust** (no Rust SDK exists; `reqwest` + `serde` on the documented JSON)

```rust
#[derive(Deserialize)]
struct SystemOneResponse { answers: HashMap<String, Answer> }

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Answer {
    Choice { choice: String, probabilities: HashMap<String, f32>, confidence: f32 },
    #[serde(other)] Other,
}
```

The returned `choice` is always one of the option keys sent, so mapping it back to a term id is a hash lookup.

**Using the probabilities**

- Apply Jev's choice when its `confidence` is at least c\_min (start at 0.6, tune on the tuning set). Otherwise the lexicon's own decision stands.
- Combine evidence instead of overriding it: final score per option = Jev probability × lexicon phonetic similarity, renormalised. A term that sounds wrong can't win on context alone.
- Log every call (span, options, probabilities, confidence, latency). The evaluation (section 11) then compares Jev, the lexicon and the combined score directly.

**Operational rules**

- **Latency:** TypeSafe does not publish latency figures, so measure from your network first. Hard timeout per call (default 300 ms); on timeout, 429 or 529 the lexicon decision stands. No retries in the typing path.
- **Cost:** you pay per input token, and output tokens are free ([models](https://docs.typesafe.ai/models.md)). At about 300 input tokens per call, cost per call is negligible.
- **Privacy:** dictated text leaves your machine for these spans. The layer is opt-in, sends only the current and previous utterance, and can be switched off per project in `.voxtype/config`.
- **Offline fallback:** a local GGUF model through a llama.cpp binding, constrained by a per-request grammar to `{span, option}` pairs, behind the same `Selector` trait. Same inputs, same outputs, no network.

```rust
trait Selector {
    async fn select(&self, req: &SelectionRequest, deadline: Instant) -> Option<Vec<SpanChoice>>;
}
// impls: JevSelector (HTTP), LocalGrammarSelector (GGUF), NoopSelector
```

**Modes**

| Mode | Jev runs | Notes |
| --- | --- | --- |
| Push-to-talk | after the text-level pass, before typing | one call per utterance, all ambiguous spans together |
| Streaming `eou` | at end of utterance, before commit | same call, after the EOU pass |
| Streaming `lag` | no | a network call doesn't fit a 3-word hold-back; ambiguous spans keep the lexicon decision |

## Two-pass decoding (push-to-talk)

The second pass re-runs only the transducer decoder, on cached encoder frames, for the uncertain spans, with a boost list built from retrieved candidates. The encoder is most of the compute and never runs twice, which is what keeps pass 2 under 15% of pass 1.

**Flow per utterance**

1. **Pass 1.** Encode the whole utterance and keep the encoder output (frames × dim, `f32`, about 1 MB per 10 s; keep it in a reused buffer). Decode with the static boost list: the top N terms by weight, N ≈ 300.
2. **Retrieve.** Run the query pipeline. If no span is uncertain, or every span was fixed with `sim ≥ 0.95`, stop here. This should be the common case.
3. **Build the dynamic boost list.** Union of the candidates for all spans, deduplicated, capped at 200 terms, plus every casing variant of each. Token sequences come pre-tokenised from the index.
4. **Pass 2.** For each uncertain span, slice the cached encoder frames from its start to end time, padded by about 0.4 s each side, and re-decode with static ∪ dynamic boosts. Spans are independent, so they decode in parallel.
5. **Splice.** Replace the pass-1 words inside each span's time range with the pass-2 words. If pass 2 is less confident than pass 1 in total, keep pass 1.
6. **Text-level fallback.** Spans still uncertain after pass 2 go through the query pipeline's select step once more, with the stricter θ.

**What transcribe.cpp needs for this.** Its public API runs encoder and decoder end to end and has no boosting, so this section depends on a patch, ideally upstreamed behind a capability flag (`supports_context_bias`), in the style the library already uses for per-family features. Without boosting a second pass would only repeat the first, so until the patch lands, pass 2 is skipped and the text-level and LLM layers carry the work.

1. **Context bias list** in the run parameters: token-id sequences plus a bonus. Inside, a token-level Aho-Corasick trie (a "context graph") gives each state an accumulated bonus. In the RNNT/TDT greedy loop, a token that extends a trie path gets the bonus added to its log-probability; on a failed extension the partial bonus is subtracted and the failure link followed, so half-matched terms are not rewarded. A graph for 500 terms builds in tens of microseconds, so it is rebuilt per call.
2. **Bias updates while streaming:** set a new bias list between stream feeds, applied from the next chunk on. This is what the rolling boost set (section 9) uses.
3. **Encoder split:** `encode` returning a handle to the encoder output, and `decode_range(handle, from_frame, to_frame, params)`. This makes pass 2 decoder-only. Before this step lands, pass 2 can re-run on the padded audio slice of each span instead: the encoder runs again, but only on 1–2 s of audio.
4. **Optional small beam** (width 4) for pass-2 spans, so a boosted path can overtake a common word that was ahead by a few tokens. Greedy with biasing already works; beam is a later refinement.

**Pronunciation overrides:** a user variant can also carry an explicit token mapping (the tokens the model actually emits → the canonical term). The splice step applies these directly, which handles terms the acoustic model never spells right however hard they are boosted.

## Nemotron streaming integration

In streaming mode, retrieval runs on each newly completed word, words are held back three words before typing so corrections can still land, and retrieved terms feed a rolling boost list that biases the decoder for the chunks that follow.

**Threads and channels**

- **ASR thread** (existing): feeds audio chunks to the cache-aware encoder and decoder, and emits word events.
- **Lexicon worker:** receives word events over a lock-free SPSC channel (`crossbeam` or `rtrb`), runs retrieval, and sends commit events to the typing thread.
- **Typing thread** (existing): types committed text.
- If the lexicon worker ever falls behind, words pass through uncorrected rather than blocking the ASR thread.

**Word states**

- *Uncommitted:* text transcribe.cpp may still revise (its stream updates carry `result_changed` and a `revision`).
- *Stable:* text transcribe.cpp has committed (`transcribe_stream_n_committed_*`, using its `STABLE_PREFIX` commit policy). The lexicon treats this as final ASR output.
- *Typed:* sent to the typing thread. Typed text is hard to take back, so all correction happens before this point.

**Per completed word**

1. When a word becomes stable, run the confidence gate and windowing on the trailing five stable words, only for windows that include the new word. Windows run over the word stream, not over chunks, so a term spoken across a chunk boundary is still found.
2. Apply text-level fixes with a stricter threshold (θ\_stream ≈ 0.88), because there is no second pass yet.
3. Add strong candidates and accepted hits to the rolling boost set.
4. Commit words that are now more than H = 3 words behind the newest one. H equals the maximum window length, so a word is never typed while it could still be part of a correction.

**Rolling boost set**

- Static top-N terms plus a session LRU of recently seen terms (TTL about 5 minutes, cap 200). Developers repeat their jargon, so a term heard once is likely to come back.
- At each chunk boundary, the lexicon worker publishes a new bias list (section 8, patch item 2) if the set changed. The ASR thread picks it up with one atomic pointer swap (`ArcSwap`) before decoding the next chunk. Rebuilding takes microseconds, so it never delays a chunk.

**End-of-utterance re-decode**

- With the encoder split (section 8, patch item 3), the ASR thread keeps a ring buffer of streamed encoder frames (last 30 s). Before that, it keeps the last 30 s of audio and re-runs each span's slice.
- At end of utterance (Nemotron's EOU signal), any span still uncertain and not yet committed is re-decoded from those frames with the current boost set, as in the two-pass flow, then committed.
- The streamed encoder frames were computed with limited right context, so this pass cannot beat a full offline pass. It still gets the boost list the first pass lacked, for a few milliseconds of decoder time.

**Commit modes** (config `lexicon.streaming.commit`)

| Mode | Behaviour | Use when |
| --- | --- | --- |
| `lag` (default) | type words H = 3 words behind, correct within the hold-back | responsive typing matters |
| `eou` | type the whole utterance at end of utterance, after the re-decode | accuracy matters more than seeing words appear |
| `off` | type immediately, no correction | lowest latency, for debugging |

**Latency budget:** retrieval per completed word p99 < 100 µs. Added perceived latency in `lag` mode is the duration of three words (roughly 1 s of speech); in `eou` mode it is the re-decode, a few milliseconds after end of utterance.

## Performance plan

Every design choice that affects speed is settled by a benchmark on 10k, 100k and 1M-variant lexicons, and CI fails on a p99 regression above 10%.

**Benchmarks** (`criterion`, in `voxtype-lexicon/benches`)

- `open`: map the index and touch the header, cold and warm page cache.
- `key_lookup`, `qgram_search`, `verify`: each layer alone, on query sets of exact, 1-edit, 2-edit and unrelated phoneme strings.
- `utterance`: the full query pipeline on recorded word streams with real confidences.
- `context_graph_build`: 50, 200 and 500 terms.
- `build`: index build time and peak memory at 1M variants.

**Profiling:** `cargo flamegraph` and `perf stat` (cache misses, branch misses) on the `utterance` bench. A `dhat` run checks the zero-allocation rule in the hot path.

**Decisions to settle by measurement**

| Question | Default | Alternative to measure |
| --- | --- | --- |
| Gram length | q = 3 (4,096 slots) | q = 4 (65,536 slots): shorter lists, more slots |
| Stop-gram cut-off | df > 5% | 2% or 10%, recall vs speed |
| Prefilter before weighted DP | none | Myers bit-parallel on `u64` |
| Posting storage | raw `u32` | delta + varint or bitpacked blocks, if memory matters more than nanoseconds |
| Key table | open addressing | `fst::Map`, smaller but slower |
| Counter scratch | stamps + `u16` counts | sorted-list merge, for very small lexicons |

**Rules for the hot path**

- Sections are read as slices from the map; no deserialisation, no copies.
- Fixed-size stack buffers (`SmallVec<[u8; 32]>`, `[u16; 64]` DP rows); no `String` or `Vec` growth per query.
- Top-k kept in a fixed array with insertion sort (k = 8); faster than a heap at this size.
- Release profile: `lto = "fat"`, `codegen-units = 1`, `panic = "abort"` for the binary; `target-cpu` left generic for distribution packages, with an opt-in native build.
- Optional `madvise(WILLNEED)` on the q-gram offsets and key table at open, so the first query doesn't stall on page faults.

**What not to optimise yet:** explicit SIMD. The count loop and DP are short and branch-light, so autovectorisation should be enough. Revisit only if the flame graph says otherwise.

## Evaluation and test plan

Success is measured as domain-term error rate and false-correction rate on your own speech, not just overall WER, because a few jargon words carry most of the value and most of the risk.

**Test sets**

- **Real:** 30–60 minutes of your own dictation about your normal work, hand-transcribed. Split 50/50 into a tuning set (thresholds) and a held-out set (reported numbers).
- **Synthetic:** each term spoken by TTS in 3–5 carrier sentences ("then run X on the cluster"), with several voices and speeds. Cheap, covers the whole lexicon, but only a proxy.

**Metrics**

| Metric | Meaning |
| --- | --- |
| Term recall | share of spoken lexicon terms that come out correctly spelled |
| False correction rate | replacements that were wrong, per 100 words |
| WER | overall, to catch damage outside terms |
| Latency | p50/p99 per stage, from the benchmarks and from live traces |

**Ablation ladder** (each step on and off): no lexicon → static boost only → text-level fix → two-pass → streaming with rolling boost. Each step must improve term recall without raising false corrections above 0.2 per 100 words.

**Learning from errors** (the loop that improves the index over time)

1. Run the synthetic set through Nemotron and align output to references.
2. Every misrecognised term gives a pair (what the model wrote, the term). Convert the output to phonemes and add it as a variant with source `observed`.
3. Count aligned phoneme substitutions across all pairs and refit the 40×40 substitution cost table from them.
4. Rebuild the index. Repeat after lexicon changes.

**Correctness tests**

- Unit tests per layer, with hand-made cases (`cube control` → `kubectl`, `engine x` → `nginx`, `get` stays `get` without context).
- Property tests (`proptest`): q-gram search returns every variant within edit distance k that a brute-force scan returns.
- Round-trip tests on the file format, including a stale tokenizer hash refusing to load.
- Fuzzing (`cargo fuzz`) of `open` on corrupt files, since the index is mapped and cast without parsing.

## Milestones and open questions

### Implementation Status (October 10, 2026)

Performance numbers below remain measurement goals, not acceptance gates,
per the implementation follow-up. No latency or accuracy target is claimed.

- M0: offline generator, cached process-pool G2P, explicit forms, identifier
  and acronym variants, query-dictionary export, atomic TSV output, and tests
  implemented. The user's 30k-term list was not supplied for spot checks.
- M1: workspace crate, validated mmap format, atomic builder, exact keys,
  q-gram postings, weighted banded verification, reusable retrieval scratch,
  scale benchmarks, and correctness/property tests implemented. Open currently
  validates all sections and dictionary entries; the pipeline still allocates.
  Parallel Rust building, query-result LRU, pre-tokenized boost tokens, and a
  dedicated fuzz runner are not implemented. Random-byte open tests exist.
- M2: opt-in Nemotron batch/final correction, confidence gating, 1-3-word
  windows, interval selection, punctuation retention, and context guards
  implemented. Held-out accuracy evaluation requires recordings and references.
- M3: explicit project selection, suppressions, spelling/weight overrides,
  between-utterance reload, and opt-in typed Jev Choice requests with timeout
  fallback implemented. Local GGUF selection now uses an optional resident
  llama.cpp helper with per-request option-ID grammar, deadlines, and offline
  fallback. Term-description rubrics remain unimplemented. Jev tests use local
  fixtures, not a paid API. Real GGUF grammar/transport passed with a small
  instruction model; selection accuracy is not certified.
- M4: blocked on native context-bias and encoder/decode-range APIs absent from
  pinned transcribe.cpp 0.3.1. No second pass or decoder boosts are attempted.
  No upstream dependency source was modified.
- M5: stable-prefix lag/eou/off commits, three-complete-word hold-back,
  correction-boundary safety, and toggle-hotkey safety implemented. `eou`
  currently means recording finalization. Corrections run in the existing ASR
  worker with nonblocking corrector locking, not a separate SPSC worker.
  Rolling decoder boosts and EOU acoustic re-decoding depend on M4.
- M6: synthetic carrier audio generation (espeak-ng), JSONL manifests,
  carrier-aligned evaluation with WER/term recall, reviewed-pair import,
  deduplicated observed variants, and embedded learned substitution costs
  implemented. Opt-in automatic project detection supports Hyprland, Sway,
  and X11 with shell-child cwd discovery and explicit-selection priority.
  Unsupported Wayland compositors and ambiguous terminals need explicit
  selection. Live TTS/evaluation validation needs espeak-ng installed; no
  held-out accuracy or false-correction target is claimed.

Build in seven steps, each shippable on its own; text-level correction alone (M2) should already give most of the term-recall gain.

1. **M0 · Lexicon and G2P.** `terms.tsv` format, `tools/lexicon_g2p.py`, query dictionary export. Done when your 30k terms convert with sensible variants on spot checks.
2. **M1 · Index and query library.** `voxtype-lexicon` crate: builder, `.vxl` format, three layers, `Scratch`, benches and property tests. Done when 100k variants meet p99 < 100 µs per window.
3. **M2 · Text-level correction in push-to-talk.** Gate → windows → select → emit on the existing word confidences, behind `lexicon.enabled`. Done when held-out term recall improves with false corrections ≤ 0.2 per 100 words.
4. **M3 · Project lexicons and LLM selection.** `LexiconSet` merge, explicit project switching, Jev Choice selection with timeout fallback. Done when the LLM beats the lexicon's own decision on ambiguous spans in the tuning set.
5. **M4 · transcribe.cpp patch and two-pass.** Context bias list, bias updates while streaming, encoder split; then pass 2 on spans. Done when pass 2 costs < 15% of pass 1 and beats M3 on term recall.
6. **M5 · Streaming.** Lexicon worker on stable words, hold-back, rolling boost set, EOU pass, commit modes. Done when streaming term recall is within a few points of push-to-talk at the latency budget.
7. **M6 · Learning loop and automatic project detection.** Synthetic carrier-sentence set, `observed` variants, learned substitution costs, focused-window project detection.

M2 and M3 need no change to transcribe.cpp, so most of the gain arrives before the patch.

