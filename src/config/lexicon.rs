use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[path = "lexicon_project.rs"]
mod project_detection;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct LexiconConfig {
    pub enabled: bool,
    pub index: Option<PathBuf>,
    pub terms: Option<PathBuf>,
    pub project: Option<PathBuf>,
    pub auto_project: bool,
    pub confidence_threshold: f32,
    pub capitalized_confidence_threshold: Option<f32>,
    pub similarity_threshold: f32,
    pub jev: JevConfig,
    pub local: LocalSelectorConfig,
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

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct LocalSelectorConfig {
    pub enabled: bool,
    pub model: Option<PathBuf>,
    pub helper: Option<PathBuf>,
    pub timeout_ms: u64,
    pub threads: u16,
}

impl Default for LocalSelectorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: None,
            helper: None,
            timeout_ms: 1000,
            threads: 4,
        }
    }
}

impl Default for LexiconConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            index: None,
            terms: None,
            project: None,
            auto_project: false,
            confidence_threshold: 0.89,
            capitalized_confidence_threshold: None,
            similarity_threshold: 0.82,
            jev: JevConfig::default(),
            local: LocalSelectorConfig::default(),
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
            Ok(_) => Ok(self.auto_project.then(project_detection::detect).flatten()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(self.auto_project.then(project_detection::detect).flatten())
            }
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
        if self
            .capitalized_confidence_threshold
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err("lexicon.capitalized_confidence_threshold must be between 0 and 1".into());
        }
        if !(1..=10_000).contains(&self.local.timeout_ms)
            || !(1..=256).contains(&self.local.threads)
            || (self.local.enabled && self.local.model.is_none())
        {
            return Err("lexicon.local requires a GGUF model, timeout_ms in 1..=10000, and threads in 1..=256".into());
        }
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
        assert!(!config.lexicon.auto_project);
        assert!(!config.lexicon.local.enabled);
        let explicit = LexiconConfig {
            project: Some("/explicit/project".into()),
            auto_project: true,
            ..Default::default()
        };
        assert_eq!(explicit.active_project().unwrap(), explicit.project);
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
