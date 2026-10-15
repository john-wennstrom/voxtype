use clap::Subcommand;
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum LexiconAction {
    /// Generate carrier-sentence WAVs and a JSONL manifest with espeak-ng
    Synthesize {
        /// Source terms.tsv, including optional spoken forms
        terms: PathBuf,
        /// Fresh directory for generated audio and manifest.jsonl
        #[arg(long)]
        output: PathBuf,
        /// TTS voices (comma-separated); each term gets three carriers/speeds
        #[arg(long, value_delimiter = ',', default_value = "en-us,en-gb,en-sc")]
        voices: Vec<String>,
    },
    /// Transcribe a synthetic manifest, report errors, and export pairs for review
    Evaluate {
        manifest: PathBuf,
        /// Recognized-text TAB canonical-term pairs (review before learning)
        #[arg(long)]
        corrections: PathBuf,
    },
    /// Compile offline ARPAbet TSV files into an atomic memory-mapped index
    Build {
        /// Pronunciation variants produced by tools/lexicon_g2p.py
        variants: PathBuf,
        /// Word-to-phoneme TSV exported by --export-query-dict
        #[arg(long)]
        query_dict: PathBuf,
        /// Destination index (default: user lexicon cache)
        #[arg(long)]
        output: Option<PathBuf>,
        /// Build into this project's cache instead of the user index
        #[arg(long, conflicts_with = "output")]
        project: Option<PathBuf>,
        /// Reviewed recognized-text TAB canonical-term pairs; learn variants and costs
        #[arg(long)]
        corrections: Option<PathBuf>,
    },
    /// Select a project's .voxtype/terms.tsv lexicon for subsequent utterances
    Use {
        /// Project directory (must already have a built index)
        #[arg(required_unless_present = "clear")]
        path: Option<PathBuf>,
        /// Return to the user lexicon only
        #[arg(long, conflicts_with = "path")]
        clear: bool,
    },
    /// Inspect candidates for a stress-free ARPAbet pronunciation
    Query {
        /// Space-separated ARPAbet phones, for example "K Y UW B K AH N T R OW L"
        phones: String,
        /// Index to query (default: configured user index)
        #[arg(long)]
        index: Option<PathBuf>,
    },
}
