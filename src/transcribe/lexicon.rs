use crate::config::LexiconConfig;
use crate::error::TranscribeError;
use crate::transcribe::WordConfidence;
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Instant, SystemTime};
use voxtype_lexicon::query::{QueryEngine, QueryOpts, Word};
use voxtype_lexicon::PhoneticIndex;

pub(super) struct Corrector {
    engine: QueryEngine,
    settings: LexiconConfig,
    path: PathBuf,
    modified: Option<SystemTime>,
    recent_hit: Option<Instant>,
    project_stamp: Option<(Option<PathBuf>, Option<SystemTime>, Option<SystemTime>)>,
    selector: Option<super::lexicon_select::JevSelector>,
    previous: String,
}

impl Corrector {
    pub fn commit_mode(&self) -> voxtype_lexicon::stream::CommitMode {
        self.settings.streaming.commit
    }

    pub fn stream_query(&mut self, words: &[Word<'_>]) -> voxtype_lexicon::query::QueryResult {
        let result = self.preview_query(words);
        if !result.replacements.is_empty() {
            self.recent_hit = Some(Instant::now());
        }
        result
    }

    pub fn preview_query(&mut self, words: &[Word<'_>]) -> voxtype_lexicon::query::QueryResult {
        let result = self.engine.query(
            words,
            QueryOpts {
                confidence_threshold: self.settings.confidence_threshold,
                similarity_threshold: self.settings.similarity_threshold.max(0.88),
                recent_context: self
                    .recent_hit
                    .is_some_and(|hit| hit.elapsed().as_secs() < 300),
            },
        );
        log_query(words, &result, self.settings.confidence_threshold);
        result
    }

    pub fn open(settings: &LexiconConfig) -> Result<Option<Self>, TranscribeError> {
        if !settings.enabled {
            return Ok(None);
        }
        settings.validate().map_err(TranscribeError::ConfigError)?;
        let path = settings
            .index_path()
            .map_err(TranscribeError::ConfigError)?;
        let index = PhoneticIndex::open(&path).map_err(|error| TranscribeError::ConfigError(format!("Cannot load lexicon: {error:#}. Run `voxtype lexicon build` or disable lexicon.enabled")))?;
        let modified = std::fs::metadata(&path)
            .and_then(|metadata| metadata.modified())
            .ok();
        let selector = super::lexicon_select::JevSelector::new(&settings.jev)?;
        let mut corrector = Self {
            engine: QueryEngine::new(index),
            settings: settings.clone(),
            path,
            modified,
            recent_hit: None,
            project_stamp: None,
            selector,
            previous: String::new(),
        };
        corrector.refresh_project()?;
        tracing::info!(index = %corrector.path.display(), confidence_threshold_percent = settings.confidence_threshold * 100.0, "Lexicon enabled (confidence scale: 0..1; popup scale: 0..100)");
        Ok(Some(corrector))
    }

    fn refresh_project(&mut self) -> Result<(), TranscribeError> {
        let project = self
            .settings
            .active_project()
            .map_err(TranscribeError::ConfigError)?;
        let index_path = project
            .as_deref()
            .map(LexiconConfig::project_index_path)
            .transpose()
            .map_err(TranscribeError::ConfigError)?;
        let index_modified = index_path.as_ref().and_then(|path| {
            std::fs::metadata(path)
                .and_then(|metadata| metadata.modified())
                .ok()
        });
        let source_modified = project.as_ref().and_then(|path| {
            std::fs::metadata(path.join(".voxtype/terms.tsv"))
                .and_then(|metadata| metadata.modified())
                .ok()
        });
        let stamp = (project.clone(), index_modified, source_modified);
        if self.project_stamp.as_ref() == Some(&stamp) {
            return Ok(());
        }
        let (index, suppressions) = if let (Some(project), Some(index_path)) =
            (&project, &index_path)
        {
            let source =
                std::fs::read_to_string(project.join(".voxtype/terms.tsv")).map_err(|error| {
                    TranscribeError::ConfigError(format!("Cannot read project terms: {error}"))
                })?;
            let suppressions: HashSet<_> = source
                .lines()
                .filter_map(|line| {
                    line.split('\t')
                        .next()?
                        .trim()
                        .strip_prefix('!')
                        .map(|term| term.trim().to_lowercase())
                })
                .collect();
            let index = PhoneticIndex::open(index_path).map_err(|error| TranscribeError::ConfigError(format!("Cannot load project lexicon: {error:#}. Build it with `voxtype lexicon build --project <path>`")))?;
            (Some(index), suppressions)
        } else {
            (None, HashSet::new())
        };
        self.engine.set_project(index, &suppressions);
        self.project_stamp = Some(stamp);
        self.recent_hit = None;
        self.previous.clear();
        Ok(())
    }

    pub fn refresh(&mut self) {
        let modified = std::fs::metadata(&self.path)
            .and_then(|metadata| metadata.modified())
            .ok();
        if modified.is_some() && modified != self.modified {
            match PhoneticIndex::open(&self.path) {
                Ok(index) => {
                    let previous_engine =
                        std::mem::replace(&mut self.engine, QueryEngine::new(index));
                    let previous_stamp = self.project_stamp.take();
                    match self.refresh_project() {
                        Ok(()) => self.modified = modified,
                        Err(error) => {
                            self.engine = previous_engine;
                            self.project_stamp = previous_stamp;
                            tracing::warn!(%error, "Keeping previous lexicon set after failed reload");
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, "Keeping previous lexicon after failed reload")
                }
            }
        }
        if let Err(error) = self.refresh_project() {
            tracing::warn!(%error, "Keeping previous project lexicon after failed reload");
        }
    }

    pub fn correct(
        &mut self,
        text: &str,
        words: &[WordConfidence],
    ) -> (String, Vec<WordConfidence>) {
        self.refresh();
        if words.is_empty()
            || !words
                .iter()
                .map(|word| word.text.as_str())
                .eq(text.split_whitespace())
        {
            tracing::warn!(
                word_count = words.len(),
                "Lexicon lookup skipped: missing or mismatched Nemotron word scores"
            );
            self.previous = text.to_string();
            return (text.to_string(), words.to_vec());
        }
        let original_words = words;
        let words: Vec<_> = words
            .iter()
            .map(|word| Word {
                text: &word.text,
                confidence: word.confidence,
            })
            .collect();
        let mut result = self.engine.query(
            &words,
            QueryOpts {
                confidence_threshold: self.settings.confidence_threshold,
                similarity_threshold: self.settings.similarity_threshold,
                recent_context: self
                    .recent_hit
                    .is_some_and(|hit| hit.elapsed().as_secs() < 300),
            },
        );
        let project_allows_remote = match self.settings.active_project() {
            Ok(None) => true,
            Err(_) => false,
            Ok(Some(project)) => match std::fs::read_to_string(project.join(".voxtype/config")) {
                Ok(contents) => {
                    toml::from_str::<toml::Value>(&contents)
                        .ok()
                        .is_some_and(|value| {
                            value
                                .get("lexicon")
                                .and_then(|value| value.get("jev"))
                                .and_then(|value| value.get("enabled"))
                                .and_then(toml::Value::as_bool)
                                != Some(false)
                        })
                }
                Err(error) => error.kind() == std::io::ErrorKind::NotFound,
            },
        };
        if project_allows_remote {
            if let Some(selector) = &self.selector {
                let request = super::lexicon_select::SelectionRequest {
                    previous: &self.previous,
                    text,
                    spans: &result.ambiguous,
                };
                if let Some(selected) = selector.select_blocking(
                    &request,
                    Instant::now() + std::time::Duration::from_millis(self.settings.jev.timeout_ms),
                ) {
                    result.replacements.extend(selected);
                    result
                        .replacements
                        .sort_by_key(|replacement| replacement.start);
                    result.text = voxtype_lexicon::query::render(&words, &result.replacements);
                }
            }
        }
        log_query(&words, &result, self.settings.confidence_threshold);
        let output = if result.replacements.is_empty() {
            text.to_string()
        } else {
            self.recent_hit = Some(Instant::now());
            result.text
        };
        self.previous.clone_from(&output);
        (
            output,
            replacement_words(original_words, &result.replacements),
        )
    }
}

fn log_query(words: &[Word<'_>], result: &voxtype_lexicon::query::QueryResult, threshold: f32) {
    let scores = words
        .iter()
        .map(|word| {
            let score = word
                .confidence
                .filter(|score| score.is_finite())
                .map_or_else(
                    || "unavailable".to_string(),
                    |score| format!("{:.2}%", score * 100.0),
                );
            format!("{}={score}", word.text)
        })
        .collect::<Vec<_>>()
        .join(" ");
    tracing::info!(%scores, threshold_percent = threshold * 100.0, "Lexicon confidence evaluation");
    for &position in &result.lookup_words {
        let word = &words[position];
        let confidence = word
            .confidence
            .filter(|score| score.is_finite())
            .map_or_else(
                || "unavailable".to_string(),
                |score| format!("{:.2}%", score * 100.0),
            );
        tracing::info!(word = word.text, %confidence, threshold_percent = threshold * 100.0, "Lexicon lookup");
    }
    for replacement in &result.replacements {
        let original = words[replacement.start..replacement.end]
            .iter()
            .map(|word| word.text)
            .collect::<Vec<_>>()
            .join(" ");
        tracing::info!(%original, picked = %replacement.canonical, score_percent = replacement.score * 100.0, "Lexicon selected (blue underline in preview)");
    }
    for span in &result.ambiguous {
        let candidates = span
            .choices
            .iter()
            .map(|choice| choice.canonical.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        tracing::info!(original = %span.original, %candidates, guarded = span.guarded, "Lexicon candidates retained without automatic replacement");
    }
    if !result.lookup_words.is_empty() {
        tracing::info!(
            evaluated_words = result.lookup_words.len(),
            selected_spans = result.replacements.len(),
            "Lexicon lookup complete"
        );
    }
}

pub(super) fn replacement_words(
    words: &[WordConfidence],
    replacements: &[voxtype_lexicon::query::Replacement],
) -> Vec<WordConfidence> {
    let mut output = Vec::new();
    let mut position = 0;
    while position < words.len() {
        if let Some(replacement) = replacements
            .iter()
            .find(|replacement| replacement.start == position && replacement.end <= words.len())
        {
            let source: Vec<_> = words[position..replacement.end]
                .iter()
                .map(|word| Word {
                    text: &word.text,
                    confidence: word.confidence,
                })
                .collect();
            let replacement = voxtype_lexicon::query::Replacement {
                start: 0,
                end: source.len(),
                ..replacement.clone()
            };
            let text = voxtype_lexicon::query::render(&source, &[replacement]);
            output.extend(text.split_whitespace().map(|text| WordConfidence {
                text: text.to_string(),
                confidence: None,
                lexicon: true,
            }));
            position += source.len();
        } else {
            output.push(words[position].clone());
            position += 1;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_evaluates_percent_gate_and_marks_picks_without_committing_context() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("preview.vxl");
        voxtype_lexicon::build::write(
            vec![voxtype_lexicon::build::Entry {
                canonical: "kubectl".into(),
                phones: voxtype_lexicon::alphabet::parse("K Y UW B K AH N T R OW L").unwrap(),
                source: 0,
                weight: 1.0,
            }],
            [("cube", "K Y UW B"), ("control", "K AH N T R OW L")]
                .into_iter()
                .map(|(word, phones)| {
                    (
                        word.into(),
                        voxtype_lexicon::alphabet::parse(phones).unwrap(),
                    )
                })
                .collect(),
            &path,
            0,
        )
        .unwrap();
        let mut corrector = Corrector {
            engine: QueryEngine::new(PhoneticIndex::open(&path).unwrap()),
            settings: LexiconConfig::default(),
            path,
            modified: None,
            recent_hit: None,
            project_stamp: None,
            selector: None,
            previous: String::new(),
        };
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_test_writer()
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            for confidence in [0.88, 0.89] {
                let words: Vec<_> = ["cube", "control"]
                    .into_iter()
                    .map(|text| WordConfidence {
                        text: text.into(),
                        confidence: Some(confidence),
                        lexicon: false,
                    })
                    .collect();
                let query_words: Vec<_> = words
                    .iter()
                    .map(|word| Word {
                        text: &word.text,
                        confidence: word.confidence,
                    })
                    .collect();
                let result = corrector.preview_query(&query_words);
                let preview = replacement_words(&words, &result.replacements);
                if confidence < 0.89 {
                    assert_eq!(result.lookup_words, vec![0, 1]);
                    assert_eq!(result.text, "kubectl");
                    assert_eq!(preview.len(), 1);
                    assert!(preview[0].lexicon);
                } else {
                    assert!(result.lookup_words.is_empty());
                    assert_eq!(preview, words);
                }
                assert!(corrector.recent_hit.is_none());
                assert!(corrector.previous.is_empty());
            }
        });
    }

    #[test]
    fn replacement_provenance_preserves_punctuation_and_unmodified_scores() {
        let words: Vec<_> = ["run", "(cube", "control),", "now"]
            .into_iter()
            .map(|text| WordConfidence {
                text: text.into(),
                confidence: Some(0.88),
                lexicon: false,
            })
            .collect();
        for canonical in ["kubectl", "kube control"] {
            let replacements = [voxtype_lexicon::query::Replacement {
                start: 1,
                end: 3,
                canonical: canonical.into(),
                score: 1.0,
                gain: 0.12,
            }];
            let picked = replacement_words(&words, &replacements);
            assert_eq!(picked.first(), words.first());
            assert_eq!(picked.last(), words.last());
            assert!(picked[1..picked.len() - 1]
                .iter()
                .all(|word| word.lexicon && word.confidence.is_none()));
            assert_eq!(
                picked
                    .iter()
                    .map(|word| word.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
                format!("run ({canonical}), now")
            );
        }
    }
}
