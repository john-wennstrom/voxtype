use clap::Subcommand;
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum LexiconAction {
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
