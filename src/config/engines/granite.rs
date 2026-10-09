use serde::{Deserialize, Serialize};

/// Native Granite TurboCTC GGUF inference settings.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct GraniteConfig {
    /// GGUF model filename or explicit path.
    pub model: String,
    /// Compute backend: auto, cpu, or cuda.
    pub backend: String,
}

impl Default for GraniteConfig {
    fn default() -> Self {
        Self {
            model: "granite-speech-5.0-470m-turboctc-Q8_0.gguf".to_string(),
            backend: "auto".to_string(),
        }
    }
}
