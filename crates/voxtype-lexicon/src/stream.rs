use crate::query::{render, QueryResult, Word};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CommitMode {
    #[default]
    Lag,
    Eou,
    Off,
}

pub struct CommitBuffer {
    mode: CommitMode,
    raw_typed: Vec<String>,
}

impl CommitBuffer {
    pub fn new(mode: CommitMode) -> Self {
        Self {
            mode,
            raw_typed: Vec::new(),
        }
    }

    pub fn pending<'words, 'text>(
        &self,
        stable: &'words [Word<'text>],
    ) -> Option<&'words [Word<'text>]> {
        if stable.len() < self.raw_typed.len()
            || !stable
                .iter()
                .zip(&self.raw_typed)
                .all(|(word, typed)| word.text == typed)
        {
            return None;
        }
        Some(&stable[self.raw_typed.len()..])
    }

    pub fn commit(
        &mut self,
        stable: &[Word<'_>],
        result: &QueryResult,
        finalizing: bool,
    ) -> Option<String> {
        let pending = self.pending(stable)?;
        let mut count = match (self.mode, finalizing) {
            (_, true) | (CommitMode::Off, false) => pending.len(),
            (CommitMode::Lag, false) => pending.len().saturating_sub(3),
            (CommitMode::Eou, false) => 0,
        };
        for replacement in &result.replacements {
            if replacement.start < count && replacement.end > count {
                count = replacement.start;
            }
        }
        if count == 0 {
            return None;
        }
        let text = render(&pending[..count], &result.replacements);
        self.raw_typed
            .extend(pending[..count].iter().map(|word| word.text.to_string()));
        Some(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::Replacement;

    #[test]
    fn holdback_never_splits_a_replacement_or_retypes_words() {
        let stable: Vec<_> = ["run", "cube", "control", "apply", "now"]
            .into_iter()
            .map(|text| Word {
                text,
                confidence: Some(0.4),
            })
            .collect();
        let result = QueryResult {
            text: "run kubectl apply now".into(),
            lookup_words: Vec::new(),
            ambiguous: Vec::new(),
            replacements: vec![Replacement {
                start: 1,
                end: 3,
                canonical: "kubectl".into(),
                score: 1.0,
                gain: 0.6,
            }],
        };
        let mut buffer = CommitBuffer::new(CommitMode::Lag);
        assert_eq!(
            buffer.commit(&stable, &result, false).as_deref(),
            Some("run")
        );
        let tail = QueryResult {
            text: "kubectl apply now".into(),
            lookup_words: Vec::new(),
            ambiguous: Vec::new(),
            replacements: vec![Replacement {
                start: 0,
                end: 2,
                ..result.replacements[0].clone()
            }],
        };
        assert_eq!(buffer.commit(&stable, &tail, false), None);
        assert_eq!(
            buffer.commit(&stable, &tail, true).as_deref(),
            Some("kubectl apply now")
        );
        assert_eq!(
            buffer.commit(
                &stable,
                &QueryResult {
                    text: String::new(),
                    lookup_words: Vec::new(),
                    replacements: Vec::new(),
                    ambiguous: Vec::new()
                },
                true
            ),
            None
        );
        let revised = [Word {
            text: "wrong",
            confidence: Some(0.1),
        }];
        assert!(buffer.pending(&revised).is_none());
    }

    #[test]
    fn eou_and_off_commit_rules() {
        let stable = [Word {
            text: "hello",
            confidence: Some(1.0),
        }];
        let result = QueryResult {
            text: "hello".into(),
            lookup_words: Vec::new(),
            replacements: Vec::new(),
            ambiguous: Vec::new(),
        };
        let mut eou = CommitBuffer::new(CommitMode::Eou);
        assert_eq!(eou.commit(&stable, &result, false), None);
        assert_eq!(eou.commit(&stable, &result, true).as_deref(), Some("hello"));
        let mut off = CommitBuffer::new(CommitMode::Off);
        assert_eq!(
            off.commit(&stable, &result, false).as_deref(),
            Some("hello")
        );
    }
}
