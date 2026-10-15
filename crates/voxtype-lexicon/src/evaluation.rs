use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize)]
pub struct Sample {
    pub audio: PathBuf,
    pub canonical: String,
    pub spoken: String,
    pub prefix: String,
    pub suffix: String,
}

fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.trim_matches(|character: char| !character.is_alphanumeric())
                .to_lowercase()
        })
        .collect()
}

impl Sample {
    pub fn reference(&self) -> String {
        format!("{} {} {}", self.prefix, self.canonical, self.suffix)
    }

    pub fn recognized_term(&self, transcript: &str) -> Option<String> {
        let transcript = words(transcript);
        let prefix = words(&self.prefix);
        let suffix = words(&self.suffix);
        if transcript.len() <= prefix.len() + suffix.len()
            || !transcript.starts_with(&prefix)
            || !transcript.ends_with(&suffix)
        {
            return None;
        }
        Some(transcript[prefix.len()..transcript.len() - suffix.len()].join(" "))
    }
}

pub fn word_errors(reference: &str, transcript: &str) -> (usize, usize) {
    let reference = words(reference);
    let transcript = words(transcript);
    let mut previous: Vec<_> = (0..=transcript.len()).collect();
    for (position, word) in reference.iter().enumerate() {
        let mut current = vec![position + 1; transcript.len() + 1];
        for (column, recognized) in transcript.iter().enumerate() {
            current[column + 1] = (previous[column + 1] + 1)
                .min(current[column] + 1)
                .min(previous[column] + usize::from(word != recognized));
        }
        previous = current;
    }
    (previous[transcript.len()], reference.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_split_terms_but_rejects_unaligned_carriers() {
        let sample = Sample {
            audio: "test.wav".into(),
            canonical: "kubectl".into(),
            spoken: "cube control".into(),
            prefix: "then run".into(),
            suffix: "on the cluster".into(),
        };
        assert_eq!(
            sample.recognized_term("Then run cube control on the cluster."),
            Some("cube control".into())
        );
        assert!(sample
            .recognized_term("then use cube control on the cluster")
            .is_none());
        assert_eq!(
            word_errors(&sample.reference(), "then run cube control on the cluster"),
            (2, 6)
        );
        assert_eq!(word_errors("git", "Git."), (0, 1));
    }
}
