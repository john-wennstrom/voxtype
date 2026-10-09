use super::Transcriber;
use crate::config::{Config, GraniteConfig};
use crate::error::TranscribeError;
use std::path::PathBuf;
use std::sync::Mutex;
use transcribe_cpp::{Backend, Model, ModelOptions, RunOptions, Session, TimestampKind};

const MAX_AUDIO_SAMPLES: usize = 16_000 * 60;

/// Resident native Granite TurboCTC transcriber.
pub struct GraniteTranscriber {
    session: Mutex<Session>,
}

impl GraniteTranscriber {
    /// Load a GGUF once and keep its inference session resident.
    pub fn new(config: &GraniteConfig) -> Result<Self, TranscribeError> {
        let backend = parse_backend(&config.backend)?;
        let path = resolve_model_path(&config.model)?;
        if !path.is_file() {
            return Err(TranscribeError::ModelNotFound(path.display().to_string()));
        }
        transcribe_cpp::init_backends_default()
            .map_err(|error| TranscribeError::InitFailed(error.to_string()))?;
        tracing::info!(model = %path.display(), "Loading native Granite model");
        let model = Model::load_with(
            &path,
            &ModelOptions {
                backend,
                device: None,
            },
        )
        .map_err(|error| TranscribeError::InitFailed(error.to_string()))?;
        tracing::info!(
            model = %model.variant(),
            backend = %model.backend(),
            "Native Granite model loaded"
        );
        let session = model
            .session()
            .map_err(|error| TranscribeError::InitFailed(error.to_string()))?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }
}

impl Transcriber for GraniteTranscriber {
    fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        validate_audio(samples)?;
        if samples.is_empty() {
            return Ok(String::new());
        }
        let mut session = self.session.lock().map_err(|_| {
            TranscribeError::InferenceFailed(
                "Granite inference session failed; restart the daemon".to_string(),
            )
        })?;
        let transcript = session
            .run(
                samples,
                &RunOptions {
                    language: Some("en".to_string()),
                    timestamps: TimestampKind::None,
                    ..Default::default()
                },
            )
            .map_err(|error| TranscribeError::InferenceFailed(error.to_string()))?;
        Ok(transcript.text.trim().to_string())
    }

    fn last_detected_language(&self) -> Option<String> {
        Some("en".to_string())
    }
}

fn parse_backend(name: &str) -> Result<Backend, TranscribeError> {
    match name {
        "auto" => Ok(Backend::Auto),
        "cpu" => Ok(Backend::Cpu),
        "cuda" if cfg!(feature = "granite-cuda") => Ok(Backend::Cuda),
        "cuda" => Err(TranscribeError::ConfigError(
            "Granite CUDA requires a build with --features granite-cuda".to_string(),
        )),
        _ => Err(TranscribeError::ConfigError(
            "granite.backend must be auto, cpu, or cuda".to_string(),
        )),
    }
}

fn resolve_model_path(model: &str) -> Result<PathBuf, TranscribeError> {
    if model.is_empty() {
        return Err(TranscribeError::ConfigError(
            "granite.model must name a GGUF file".to_string(),
        ));
    }
    if let Some(relative) = model.strip_prefix("~/") {
        return dirs::home_dir()
            .map(|home| home.join(relative))
            .ok_or_else(|| {
                TranscribeError::ConfigError("Home directory is unavailable".to_string())
            });
    }
    let path = PathBuf::from(model);
    if path.is_absolute() || path.components().count() > 1 {
        Ok(path)
    } else {
        Ok(Config::models_dir().join(path))
    }
}

fn validate_audio(samples: &[f32]) -> Result<(), TranscribeError> {
    if samples.len() > MAX_AUDIO_SAMPLES {
        return Err(TranscribeError::AudioFormat(
            "Granite accepts at most 60 seconds of 16 kHz mono audio per transcription".to_string(),
        ));
    }
    if samples
        .iter()
        .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
    {
        return Err(TranscribeError::AudioFormat(
            "Granite requires finite PCM samples in the range [-1, 1]".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn granite_backend_selection_is_explicit() {
        assert_eq!(parse_backend("auto").unwrap(), Backend::Auto);
        assert_eq!(parse_backend("cpu").unwrap(), Backend::Cpu);
        assert_eq!(
            parse_backend("cuda").is_ok(),
            cfg!(feature = "granite-cuda")
        );
        assert!(parse_backend("invalid").is_err());
    }

    #[test]
    fn granite_model_paths_preserve_explicit_paths() {
        assert_eq!(
            resolve_model_path("/tmp/model.gguf").unwrap(),
            PathBuf::from("/tmp/model.gguf")
        );
        assert_eq!(
            resolve_model_path("./model.gguf").unwrap(),
            PathBuf::from("./model.gguf")
        );
        assert_eq!(
            resolve_model_path("model.gguf").unwrap(),
            Config::models_dir().join("model.gguf")
        );
        assert!(resolve_model_path("").is_err());
    }

    #[test]
    fn granite_audio_validation_rejects_invalid_and_oversized_input() {
        assert!(validate_audio(&[]).is_ok());
        assert!(validate_audio(&[-1.0, 0.0, 1.0]).is_ok());
        for sample in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.1, -1.1] {
            assert!(validate_audio(&[sample]).is_err());
        }
        assert!(validate_audio(&vec![0.0; MAX_AUDIO_SAMPLES + 1]).is_err());
    }

    #[test]
    fn granite_transcriber_satisfies_shared_transcriber_contract() {
        fn assert_contract<BackendType: Transcriber>() {}
        assert_contract::<GraniteTranscriber>();
    }

    #[test]
    #[ignore = "requires VOXTYPE_GRANITE_MODEL pointing to installed GGUF weights"]
    fn granite_resident_native_speech_smoke() {
        let settings = GraniteConfig {
            model: std::env::var("VOXTYPE_GRANITE_MODEL").unwrap(),
            backend: std::env::var("VOXTYPE_GRANITE_BACKEND").unwrap_or_else(|_| "cpu".to_string()),
        };
        let transcriber = GraniteTranscriber::new(&settings).unwrap();
        let mut reader = hound::WavReader::open("tests/fixtures/vad/speech_hello.wav").unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 16_000);
        assert_eq!(spec.channels, 1);
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|sample| sample.unwrap() as f32 / 32768.0)
            .collect();
        let first = transcriber.transcribe(&samples).unwrap();
        let second = transcriber.transcribe(&samples).unwrap();
        assert!(
            first.to_lowercase().contains("hello"),
            "Unexpected transcript: {first}"
        );
        assert_eq!(first, second);
        assert_eq!(transcriber.last_detected_language().as_deref(), Some("en"));
    }
}
