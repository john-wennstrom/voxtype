use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct NemotronConfig {
    pub model: String,
    pub backend: String,
    pub streaming: bool,
    pub streaming_chunk_ms: u32,
}

impl Default for NemotronConfig {
    fn default() -> Self {
        Self {
            model: "nemotron-speech-streaming-en-0.6b-Q8_0.gguf".to_string(),
            backend: "auto".to_string(),
            streaming: true,
            streaming_chunk_ms: 160,
        }
    }
}

impl NemotronConfig {
    pub fn attention_right_context(&self) -> Result<i32, String> {
        match self.streaming_chunk_ms {
            80 => Ok(0),
            160 => Ok(1),
            560 => Ok(6),
            1120 => Ok(13),
            _ => Err("nemotron.streaming_chunk_ms must be 80, 160, 560, or 1120".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nemotron_defaults_enable_low_latency_streaming() {
        let config: NemotronConfig = toml::from_str("").unwrap();
        assert!(config.streaming);
        assert_eq!(config.streaming_chunk_ms, 160);
        assert_eq!(config.attention_right_context().unwrap(), 1);
        assert_eq!(config.backend, "auto");
    }

    #[test]
    fn nemotron_chunk_settings_match_trained_contexts() {
        for (chunk_ms, right_context) in [(80, 0), (160, 1), (560, 6), (1120, 13)] {
            let config = NemotronConfig {
                streaming_chunk_ms: chunk_ms,
                ..Default::default()
            };
            assert_eq!(config.attention_right_context().unwrap(), right_context);
        }
        let config = NemotronConfig {
            streaming_chunk_ms: 100,
            ..Default::default()
        };
        assert!(config.attention_right_context().is_err());
    }
}
