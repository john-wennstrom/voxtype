use crate::{Candidate, PhoneticIndex, Scratch, SearchOpts, TermRef};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};

const PROJECT: u32 = 1 << 31;

fn term<'index>(
    user: &'index PhoneticIndex,
    project: Option<&'index PhoneticIndex>,
    id: u32,
) -> TermRef<'index> {
    if id & PROJECT != 0 {
        project.expect("project candidate").term(id & !PROJECT)
    } else {
        user.term(id)
    }
}

pub struct Word<'text> {
    pub text: &'text str,
    pub confidence: Option<f32>,
}

#[derive(Clone, Copy)]
pub struct QueryOpts {
    pub confidence_threshold: f32,
    pub capitalized_confidence_threshold: Option<f32>,
    pub similarity_threshold: f32,
    pub recent_context: bool,
}

impl Default for QueryOpts {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.89,
            capitalized_confidence_threshold: None,
            similarity_threshold: 0.82,
            recent_context: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Choice {
    pub canonical: String,
    pub sim: f32,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub original: String,
    pub choices: Vec<Choice>,
    pub guarded: bool,
}

#[derive(Clone, Debug)]
pub struct Replacement {
    pub start: usize,
    pub end: usize,
    pub canonical: String,
    pub score: f32,
    pub gain: f32,
}

pub struct QueryResult {
    pub text: String,
    pub replacements: Vec<Replacement>,
    pub ambiguous: Vec<Span>,
    pub lookup_words: Vec<usize>,
}

fn common(word: &str) -> bool {
    matches!(
        word.trim_matches(|character: char| !character.is_alphabetic())
            .to_ascii_lowercase()
            .as_str(),
        "a" | "an"
            | "the"
            | "get"
            | "in"
            | "on"
            | "to"
            | "of"
            | "and"
            | "or"
            | "is"
            | "it"
            | "for"
            | "as"
            | "at"
            | "be"
            | "by"
            | "do"
            | "go"
            | "if"
            | "me"
            | "no"
            | "so"
            | "up"
            | "we"
            | "you"
    )
}

pub fn render(words: &[Word<'_>], replacements: &[Replacement]) -> String {
    let mut output = String::new();
    let mut position = 0;
    while position < words.len() {
        if !output.is_empty() {
            output.push(' ');
        }
        if let Some(replacement) = replacements
            .iter()
            .find(|replacement| replacement.start == position)
        {
            let first = words[position].text;
            let prefix_end = first
                .char_indices()
                .find(|(_, character)| !matches!(character, '(' | '[' | '{' | '"' | '\''))
                .map(|(position, _)| position)
                .unwrap_or(first.len());
            output.push_str(&first[..prefix_end]);
            output.push_str(&replacement.canonical);
            let last = words[replacement.end - 1].text;
            let suffix_start = last
                .char_indices()
                .rev()
                .find(|(_, character)| {
                    !matches!(
                        character,
                        '.' | ',' | '!' | '?' | ':' | ';' | ')' | ']' | '}' | '"' | '\''
                    )
                })
                .map(|(position, character)| position + character.len_utf8())
                .unwrap_or(0);
            output.push_str(&last[suffix_start..]);
            position = replacement.end;
        } else {
            output.push_str(words[position].text);
            position += 1;
        }
    }
    output
}

fn fallback(word: &str, output: &mut SmallVec<[u8; 32]>) {
    for character in word.bytes().map(|byte| byte.to_ascii_lowercase()) {
        let phones: &[u8] = match character {
            b'a' => &[1],
            b'e' => &[10],
            b'i' => &[16],
            b'o' => &[24],
            b'u' => &[2],
            b'b' => &[6],
            b'c' | b'k' | b'q' => &[19],
            b'd' => &[8],
            b'f' => &[13],
            b'g' => &[14],
            b'h' => &[15],
            b'j' => &[18],
            b'l' => &[20],
            b'm' => &[21],
            b'n' => &[22],
            b'p' => &[26],
            b'r' => &[27],
            b's' => &[28],
            b't' => &[30],
            b'v' => &[34],
            b'w' => &[35],
            b'x' => &[19, 28],
            b'y' => &[36],
            b'z' => &[37],
            _ => &[],
        };
        if output.len() + phones.len() > 63 {
            break;
        }
        output.extend_from_slice(phones);
    }
}

pub struct QueryEngine {
    pub index: PhoneticIndex,
    pub scratch: Scratch,
    candidates: Vec<Candidate>,
    project: Option<PhoneticIndex>,
    project_scratch: Scratch,
    project_candidates: Vec<Candidate>,
    suppressed: Vec<bool>,
    overrides: Vec<Option<u32>>,
    weights: Vec<f32>,
    aliases: HashMap<String, String>,
    suppressed_aliases: HashSet<String>,
}

impl QueryEngine {
    pub fn new(index: PhoneticIndex) -> Self {
        let scratch = Scratch::new(&index);
        Self {
            index,
            scratch,
            candidates: Vec::with_capacity(16),
            project: None,
            project_scratch: Scratch::default(),
            project_candidates: Vec::with_capacity(8),
            suppressed: Vec::new(),
            overrides: Vec::new(),
            weights: Vec::new(),
            aliases: HashMap::new(),
            suppressed_aliases: HashSet::new(),
        }
    }

    pub fn set_aliases(&mut self, aliases: HashMap<String, String>) {
        self.aliases = aliases;
    }

    pub fn set_project(&mut self, project: Option<PhoneticIndex>, suppressions: &HashSet<String>) {
        self.suppressed_aliases = suppressions
            .iter()
            .map(|term| term.to_lowercase())
            .collect();
        let project_ids: HashMap<_, _> = project
            .as_ref()
            .map(|project| {
                (0..project.term_count() as u32)
                    .map(|id| (project.term(id).canonical.to_lowercase(), id))
                    .collect()
            })
            .unwrap_or_default();
        self.suppressed = (0..self.index.term_count() as u32)
            .map(|id| suppressions.contains(&self.index.term(id).canonical.to_lowercase()))
            .collect();
        self.overrides = (0..self.index.term_count() as u32)
            .map(|id| {
                project_ids
                    .get(&self.index.term(id).canonical.to_lowercase())
                    .copied()
            })
            .collect();
        self.project_scratch = project.as_ref().map(Scratch::new).unwrap_or_default();
        self.weights = self
            .overrides
            .iter()
            .enumerate()
            .map(|(id, override_id)| {
                override_id
                    .and_then(|id| project.as_ref().map(|project| project.term(id).weight))
                    .unwrap_or_else(|| self.index.term(id as u32).weight)
            })
            .collect();
        self.project = project;
    }

    fn search(&mut self, phones: &[u8]) {
        let options = SearchOpts {
            max_edits: (phones.len() / 4).max(1),
            excluded_terms: &self.suppressed,
            term_weights: &self.weights,
            ..Default::default()
        };
        self.index
            .search(phones, &options, &mut self.scratch, &mut self.candidates);
        if let Some(project) = &self.project {
            for candidate in &mut self.candidates {
                if let Some(Some(id)) = self.overrides.get(candidate.term as usize) {
                    candidate.term = *id | PROJECT;
                }
            }
            project.search(
                phones,
                &SearchOpts {
                    max_edits: options.max_edits,
                    ..Default::default()
                },
                &mut self.project_scratch,
                &mut self.project_candidates,
            );
            for candidate in &self.project_candidates {
                let mut candidate = *candidate;
                candidate.term |= PROJECT;
                if let Some(existing) = self
                    .candidates
                    .iter_mut()
                    .find(|existing| existing.term == candidate.term)
                {
                    if candidate.sim >= existing.sim {
                        *existing = candidate;
                    }
                } else {
                    self.candidates.push(candidate);
                }
            }
            self.candidates.sort_by(|left, right| {
                right
                    .score
                    .total_cmp(&left.score)
                    .then_with(|| (right.term & PROJECT).cmp(&(left.term & PROJECT)))
            });
            self.candidates.truncate(8);
        }
    }

    pub fn query(&mut self, words: &[Word<'_>], options: QueryOpts) -> QueryResult {
        let mut phonemes = Vec::with_capacity(words.len());
        let mut uncertain = Vec::with_capacity(words.len());
        for word in words {
            let mut phones = SmallVec::new();
            let capitalized = word
                .text
                .chars()
                .find(|character| character.is_alphabetic())
                .is_some_and(char::is_uppercase);
            let threshold = if capitalized {
                options
                    .capitalized_confidence_threshold
                    .unwrap_or(options.confidence_threshold)
            } else {
                options.confidence_threshold
            };
            let known = self.index.g2p_word(word.text, &mut phones)
                || self
                    .project
                    .as_ref()
                    .is_some_and(|project| project.g2p_word(word.text, &mut phones));
            uncertain.push(
                !known
                    || word
                        .confidence
                        .is_some_and(|confidence| confidence.is_finite() && confidence < threshold),
            );
            if !known {
                fallback(word.text, &mut phones);
            }
            phonemes.push(phones);
        }
        let mut proposals = Vec::new();
        let mut ambiguous = Vec::new();
        let mut guarded = Vec::new();
        for start in 0..words.len() {
            let mut phones = SmallVec::<[u8; 32]>::new();
            for end in start + 1..=(start + 3).min(words.len()) {
                phones.extend_from_slice(&phonemes[end - 1]);
                if phones.len() > 63 {
                    break;
                }
                let original = words[start..end]
                    .iter()
                    .map(|word| word.text)
                    .collect::<Vec<_>>()
                    .join(" ");
                if let Some(canonical) = self
                    .aliases
                    .get(&crate::terms::normalize(&original))
                    .filter(|canonical| {
                        !self.suppressed_aliases.contains(&canonical.to_lowercase())
                    })
                {
                    let replacement = Replacement {
                        start,
                        end,
                        canonical: canonical.clone(),
                        score: 1.0,
                        gain: (end - start) as f32,
                    };
                    if words[start..end].iter().all(|word| common(word.text)) {
                        guarded.push(replacement);
                    } else {
                        proposals.push(replacement);
                    }
                    continue;
                }
                if !uncertain[start..end].iter().any(|uncertain| *uncertain) {
                    continue;
                }
                self.search(&phones);
                let original = words[start..end]
                    .iter()
                    .map(|word| word.text)
                    .collect::<Vec<_>>()
                    .join(" ");
                self.candidates.retain(|candidate| {
                    term(&self.index, self.project.as_ref(), candidate.term).canonical
                        != original.trim_matches(|character: char| !character.is_alphanumeric())
                });
                let Some(best) = self.candidates.first() else {
                    continue;
                };
                if best.sim < 0.65 {
                    continue;
                }
                let confidence = words[start..end]
                    .iter()
                    .map(|word| {
                        word.confidence
                            .filter(|confidence| confidence.is_finite())
                            .unwrap_or(1.0)
                            .clamp(0.0, 1.0)
                    })
                    .sum::<f32>()
                    / (end - start) as f32;
                let guard = words[start..end].iter().all(|word| common(word.text));
                let tied = self
                    .candidates
                    .get(1)
                    .is_some_and(|second| best.score - second.score < 0.05);
                if best.sim < options.similarity_threshold || guard || tied {
                    ambiguous.push(Span {
                        start,
                        end,
                        original,
                        choices: self
                            .candidates
                            .iter()
                            .take(5)
                            .map(|candidate| Choice {
                                canonical: term(&self.index, self.project.as_ref(), candidate.term)
                                    .canonical
                                    .to_string(),
                                sim: candidate.sim,
                                score: candidate.score,
                            })
                            .collect(),
                        guarded: guard,
                    });
                }
                if best.sim >= options.similarity_threshold && best.score > confidence && !tied {
                    let replacement = Replacement {
                        start,
                        end,
                        canonical: term(&self.index, self.project.as_ref(), best.term)
                            .canonical
                            .to_string(),
                        score: best.score,
                        gain: best.score - confidence,
                    };
                    if guard {
                        guarded.push(replacement);
                    } else {
                        proposals.push(replacement);
                    }
                }
            }
        }
        if options.recent_context || !proposals.is_empty() {
            proposals.extend(guarded);
        }
        let replacements = select(proposals);
        ambiguous.retain(|span| {
            !replacements
                .iter()
                .any(|replacement| replacement.start < span.end && span.start < replacement.end)
        });
        let text = render(words, &replacements);
        QueryResult {
            text,
            replacements,
            ambiguous,
            lookup_words: uncertain
                .iter()
                .enumerate()
                .filter_map(|(position, uncertain)| uncertain.then_some(position))
                .collect(),
        }
    }
}

pub fn select(mut proposals: Vec<Replacement>) -> Vec<Replacement> {
    proposals.sort_by_key(|replacement| (replacement.end, replacement.start));
    let mut best = vec![0.0f32; proposals.len() + 1];
    let mut predecessors = Vec::with_capacity(proposals.len());
    for (position, proposal) in proposals.iter().enumerate() {
        let previous = proposals[..position].partition_point(|other| other.end <= proposal.start);
        predecessors.push(previous);
        best[position + 1] = best[position].max(best[previous] + proposal.gain);
    }
    let mut selected = Vec::new();
    let mut position = proposals.len();
    while position > 0 {
        let previous = predecessors[position - 1];
        if best[previous] + proposals[position - 1].gain > best[position - 1] {
            selected.push(proposals[position - 1].clone());
            position = previous;
        } else {
            position -= 1;
        }
    }
    selected.reverse();
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        alphabet,
        build::{self, Entry},
    };
    use std::collections::BTreeMap;

    #[test]
    fn jargon_correction_and_common_word_guard() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("test.vxl");
        let entries = [("kubectl", "K Y UW B K AH N T R OW L"), ("git", "G IH T")]
            .into_iter()
            .map(|(canonical, phones)| Entry {
                canonical: canonical.into(),
                phones: alphabet::parse(phones).unwrap(),
                source: 0,
                weight: 1.0,
            })
            .collect();
        let words = [
            ("cube", "K Y UW B"),
            ("control", "K AH N T R OW L"),
            ("get", "G EH T"),
        ]
        .into_iter()
        .map(|(word, phones)| (word.into(), alphabet::parse(phones).unwrap()))
        .collect::<BTreeMap<_, _>>();
        build::write(entries, words, &path, 0).unwrap();
        let mut engine = QueryEngine::new(PhoneticIndex::open(&path).unwrap());
        for (confidence, expected) in [(0.88, "kubectl"), (0.89, "cube control")] {
            let result = engine.query(
                &[
                    Word {
                        text: "cube",
                        confidence: Some(confidence),
                    },
                    Word {
                        text: "control",
                        confidence: Some(confidence),
                    },
                ],
                QueryOpts::default(),
            );
            assert_eq!(result.text, expected);
            assert_eq!(
                result.lookup_words,
                if confidence < 0.89 {
                    vec![0, 1]
                } else {
                    Vec::new()
                }
            );
        }
        let result = engine.query(
            &[
                Word {
                    text: "cube",
                    confidence: Some(0.3),
                },
                Word {
                    text: "control,",
                    confidence: Some(0.4),
                },
            ],
            QueryOpts::default(),
        );
        assert_eq!(result.text, "kubectl,");
        assert_eq!(
            engine
                .query(
                    &[
                        Word {
                            text: "(cube",
                            confidence: Some(0.3)
                        },
                        Word {
                            text: "control),",
                            confidence: Some(0.4)
                        }
                    ],
                    QueryOpts::default()
                )
                .text,
            "(kubectl),"
        );
        let word = [Word {
            text: "get",
            confidence: Some(0.2),
        }];
        let common_word = [Word {
            text: "get",
            confidence: Some(0.88),
        }];
        let result = engine.query(&common_word, QueryOpts::default());
        assert_eq!(result.lookup_words, vec![0]);
        assert_eq!(result.text, "get");
        assert_eq!(engine.query(&word, QueryOpts::default()).text, "get");
        assert_eq!(
            engine
                .query(
                    &word,
                    QueryOpts {
                        recent_context: true,
                        ..Default::default()
                    }
                )
                .text,
            "git"
        );
        assert_eq!(
            engine
                .query(
                    &[
                        Word {
                            text: "cube",
                            confidence: Some(1.0)
                        },
                        Word {
                            text: "control",
                            confidence: Some(1.0)
                        }
                    ],
                    QueryOpts::default()
                )
                .text,
            "cube control"
        );
    }

    #[test]
    fn capitalized_words_use_an_optional_higher_lookup_gate() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("test.vxl");
        let phones = alphabet::parse("V AA K S T AY P").unwrap();
        build::write(
            vec![Entry {
                canonical: "Voxtype".into(),
                phones: phones.clone(),
                source: 0,
                weight: 1.0,
            }],
            BTreeMap::from([("vuxtype".into(), phones)]),
            &path,
            0,
        )
        .unwrap();
        let mut engine = QueryEngine::new(PhoneticIndex::open(&path).unwrap());
        for (text, confidence, expected) in [
            ("Vuxtype", 0.94, "Voxtype"),
            ("Vuxtype", 0.95, "Vuxtype"),
            ("vuxtype", 0.94, "vuxtype"),
            ("\"Vuxtype", 0.94, "\"Voxtype"),
        ] {
            let result = engine.query(
                &[Word {
                    text,
                    confidence: Some(confidence),
                }],
                QueryOpts {
                    capitalized_confidence_threshold: Some(0.95),
                    ..Default::default()
                },
            );
            assert_eq!(result.text, expected);
        }
        assert_eq!(
            engine
                .query(
                    &[Word {
                        text: "Vuxtype",
                        confidence: Some(0.94)
                    }],
                    QueryOpts::default()
                )
                .text,
            "Vuxtype"
        );
    }

    #[test]
    fn reviewed_aliases_correct_unknown_names_without_a_g2p_guess() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("test.vxl");
        build::write(
            vec![Entry {
                canonical: "Voxtype".into(),
                phones: alphabet::parse("V AA K S T AY P").unwrap(),
                source: 0,
                weight: 1.0,
            }],
            BTreeMap::new(),
            &path,
            0,
        )
        .unwrap();
        let mut engine = QueryEngine::new(PhoneticIndex::open(&path).unwrap());
        engine.set_aliases(crate::terms::aliases("Voxtype\tVuxtype|vox type\n").unwrap());
        let result = engine.query(
            &[Word {
                text: "Vuxtype,",
                confidence: Some(0.99),
            }],
            QueryOpts::default(),
        );
        assert_eq!(result.text, "Voxtype,");
        assert_eq!(result.replacements.len(), 1);
        engine.set_project(None, &HashSet::from(["voxtype".into()]));
        assert_ne!(
            engine
                .query(
                    &[Word {
                        text: "Vuxtype",
                        confidence: Some(0.99)
                    }],
                    QueryOpts::default()
                )
                .text,
            "Voxtype"
        );
        engine.set_aliases(HashMap::new());
        assert_ne!(
            engine
                .query(
                    &[Word {
                        text: "Vuxtype",
                        confidence: Some(0.99)
                    }],
                    QueryOpts::default()
                )
                .text,
            "Voxtype"
        );
    }

    #[test]
    fn interval_selection_prefers_total_gain() {
        let proposals = [(0, 2, 0.8), (0, 1, 0.5), (1, 2, 0.5)]
            .into_iter()
            .map(|(start, end, gain)| Replacement {
                start,
                end,
                canonical: "term".into(),
                score: 1.0,
                gain,
            })
            .collect();
        assert_eq!(select(proposals).len(), 2);
    }

    #[test]
    fn project_overrides_preserve_user_pronunciations_and_suppressions() {
        let directory = tempfile::tempdir().unwrap();
        let dictionary = BTreeMap::from([("cube".into(), alphabet::parse("K Y UW B").unwrap())]);
        let user_path = directory.path().join("user.vxl");
        let project_path = directory.path().join("project.vxl");
        build::write(
            vec![Entry {
                canonical: "kubectl".into(),
                phones: alphabet::parse("K Y UW B").unwrap(),
                source: 0,
                weight: 1.0,
            }],
            dictionary,
            &user_path,
            0,
        )
        .unwrap();
        build::write(
            vec![Entry {
                canonical: "KubeCtl".into(),
                phones: alphabet::parse("K AH T L").unwrap(),
                source: 0,
                weight: 0.8,
            }],
            BTreeMap::new(),
            &project_path,
            0,
        )
        .unwrap();
        let mut engine = QueryEngine::new(PhoneticIndex::open(&user_path).unwrap());
        engine.set_project(
            Some(PhoneticIndex::open(&project_path).unwrap()),
            &HashSet::new(),
        );
        let words = [Word {
            text: "cube",
            confidence: Some(0.3),
        }];
        let result = engine.query(&words, QueryOpts::default());
        assert_eq!(result.text, "KubeCtl");
        assert!((result.replacements[0].score - 0.8).abs() < 0.001);
        engine.set_project(None, &HashSet::from(["kubectl".into()]));
        assert_eq!(engine.query(&words, QueryOpts::default()).text, "cube");
    }

    #[test]
    fn suppression_is_applied_before_candidate_limit() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("user.vxl");
        let entries = (0..10)
            .map(|id| Entry {
                canonical: format!("term{id}"),
                phones: alphabet::parse("K Y UW B").unwrap(),
                source: 0,
                weight: 1.0,
            })
            .collect();
        build::write(
            entries,
            BTreeMap::from([("cube".into(), alphabet::parse("K Y UW B").unwrap())]),
            &path,
            0,
        )
        .unwrap();
        let mut engine = QueryEngine::new(PhoneticIndex::open(&path).unwrap());
        engine.set_project(None, &(0..9).map(|id| format!("term{id}")).collect());
        assert_eq!(
            engine
                .query(
                    &[Word {
                        text: "cube",
                        confidence: Some(0.2)
                    }],
                    QueryOpts::default()
                )
                .text,
            "term9"
        );
    }
}
