use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct LexiconConfig {
    pub enabled: bool,
    pub index: Option<PathBuf>,
    pub project: Option<PathBuf>,
    pub confidence_threshold: f32,
    pub similarity_threshold: f32,
    pub jev: JevConfig,
    pub streaming: LexiconStreamingConfig,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct LexiconStreamingConfig {
    pub commit: voxtype_lexicon::stream::CommitMode,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct JevConfig {
    pub enabled: bool,
    pub model: String,
    pub timeout_ms: u64,
    pub confidence_threshold: f32,
}

impl Default for JevConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "jev-latest".into(),
            timeout_ms: 300,
            confidence_threshold: 0.6,
        }
    }
}

impl Default for LexiconConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            index: None,
            project: None,
            confidence_threshold: 0.89,
            similarity_threshold: 0.82,
            jev: JevConfig::default(),
            streaming: LexiconStreamingConfig::default(),
        }
    }
}

impl LexiconConfig {
    pub fn active_project_file() -> Result<PathBuf, String> {
        dirs::config_dir()
            .map(|path| path.join("voxtype/lexicon/active-project"))
            .ok_or_else(|| "Cannot determine voxtype config directory".to_string())
    }

    pub fn active_project(&self) -> Result<Option<PathBuf>, String> {
        if let Some(project) = &self.project {
            return Ok(Some(project.clone()));
        }
        let path = Self::active_project_file()?;
        match std::fs::read_to_string(path) {
            Ok(path) if !path.trim().is_empty() => Ok(Some(PathBuf::from(path.trim()))),
            Ok(_) => Ok(None),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(format!("Cannot read active lexicon project: {error}")),
        }
    }

    pub fn project_index_path(project: &std::path::Path) -> Result<PathBuf, String> {
        use sha2::{Digest, Sha256};
        let project = project
            .canonicalize()
            .map_err(|error| format!("Cannot resolve project: {error}"))?;
        let hash = format!(
            "{:x}",
            Sha256::digest(project.as_os_str().as_encoded_bytes())
        );
        dirs::cache_dir()
            .map(|path| path.join("voxtype/lexicon").join(format!("{hash}.vxl")))
            .ok_or_else(|| "Cannot determine lexicon cache directory".to_string())
    }

    pub fn index_path(&self) -> Result<PathBuf, String> {
        self.index
            .clone()
            .or_else(|| dirs::cache_dir().map(|path| path.join("voxtype/lexicon/user.vxl")))
            .ok_or_else(|| {
                "Cannot determine lexicon cache directory; set lexicon.index".to_string()
            })
    }

    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("confidence_threshold", self.confidence_threshold),
            ("similarity_threshold", self.similarity_threshold),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!("lexicon.{name} must be between 0 and 1"));
            }
        }
        if self.jev.timeout_ms == 0
            || self.jev.timeout_ms > 10_000
            || self.jev.model.trim().is_empty()
            || !self.jev.confidence_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.jev.confidence_threshold)
        {
            return Err("lexicon.jev requires a model, timeout_ms in 1..=10000, and confidence_threshold in 0..=1".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_opt_in() {
        let config: crate::config::Config = toml::from_str("").unwrap();
        assert!(!config.lexicon.enabled);
        assert_eq!(config.lexicon.confidence_threshold, 0.89);
        let mut live = config.clone();
        live.engine = crate::config::TranscriptionEngine::Nemotron;
        live.nemotron = Some(crate::config::NemotronConfig {
            streaming: true,
            ..Default::default()
        });
        assert!(!live.streaming_active());
        live.lexicon.enabled = true;
        assert!(live.streaming_active());
        live.lexicon.streaming.commit = voxtype_lexicon::stream::CommitMode::Eou;
        assert!(!live.streaming_active());
        assert_eq!(config.lexicon.similarity_threshold, 0.82);
        let config: LexiconConfig = toml::from_str("enabled = true").unwrap();
        assert!(config.validate().is_ok());
        let config = LexiconConfig {
            confidence_threshold: f32::NAN,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
