use super::granite::resolve_model_path;
use super::{StreamHandle, StreamingEvent, StreamingTranscriber, Transcriber, WordConfidence};
use crate::config::NemotronConfig;
use crate::error::TranscribeError;
use std::sync::Mutex;
use tokio::sync::{mpsc, oneshot};
use transcribe_cpp::{
    Backend, Model, ModelOptions, ParakeetStreamOptions, RunOptions, Session, StreamExtension,
    StreamOptions, TimestampKind, Transcript,
};

pub struct NemotronTranscriber {
    model: Model,
    batch: Mutex<Session>,
    settings: NemotronConfig,
}

impl NemotronTranscriber {
    pub fn new(settings: &NemotronConfig) -> Result<Self, TranscribeError> {
        settings
            .attention_right_context()
            .map_err(TranscribeError::ConfigError)?;
        let backend = match settings.backend.as_str() {
            "auto" => Backend::Auto,
            "cpu" => Backend::Cpu,
            "cuda" if cfg!(feature = "granite-cuda") => Backend::Cuda,
            "cuda" => {
                return Err(TranscribeError::ConfigError(
                    "Nemotron CUDA requires a build with --features nemotron-cuda".to_string(),
                ));
            }
            _ => {
                return Err(TranscribeError::ConfigError(
                    "nemotron.backend must be auto, cpu, or cuda".to_string(),
                ));
            }
        };
        let path = resolve_model_path(&settings.model)?;
        if !path.is_file() {
            return Err(TranscribeError::ModelNotFound(path.display().to_string()));
        }
        transcribe_cpp::init_backends_default().map_err(native_error)?;
        let model = Model::load_with(
            &path,
            &ModelOptions {
                backend,
                device: None,
            },
        )
        .map_err(native_error)?;
        tracing::info!(model = %model.variant(), backend = %model.backend(), "Native Nemotron model loaded");
        let batch = Mutex::new(model.session().map_err(native_error)?);
        Ok(Self {
            model,
            batch,
            settings: settings.clone(),
        })
    }
}

impl Transcriber for NemotronTranscriber {
    fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        validate_samples(samples)?;
        if samples.is_empty() {
            return Ok(String::new());
        }
        let mut session = self.batch.lock().map_err(|_| {
            TranscribeError::InferenceFailed(
                "Nemotron session failed; restart the daemon".to_string(),
            )
        })?;
        session
            .run(samples, &run_options())
            .map(|transcript| transcript.text.trim().to_string())
            .map_err(native_error)
    }

    fn as_streaming(&self) -> Option<&dyn StreamingTranscriber> {
        self.settings.streaming.then_some(self)
    }

    fn last_detected_language(&self) -> Option<String> {
        Some("en".to_string())
    }
}

impl StreamingTranscriber for NemotronTranscriber {
    fn start_stream(
        &self,
        samples_rx: mpsc::Receiver<Vec<f32>>,
    ) -> Result<StreamHandle, TranscribeError> {
        let session = self.model.session().map_err(native_error)?;
        let settings = self.settings.clone();
        let (events_tx, events_rx) = mpsc::channel(64);
        let (cancel_tx, cancel_rx) = oneshot::channel();
        let worker = tokio::task::spawn_blocking(move || {
            let outcome = drive_stream(session, &settings, samples_rx, cancel_rx, &events_tx);
            if let Err(error) = outcome {
                let _ = events_tx.blocking_send(StreamingEvent::Error(error));
            }
            let _ = events_tx.blocking_send(StreamingEvent::Ended);
            Ok(())
        });
        let task = tokio::spawn(async move {
            worker.await.map_err(|error| {
                TranscribeError::InferenceFailed(format!(
                    "Nemotron streaming worker failed: {error}"
                ))
            })?
        });
        Ok(StreamHandle {
            events: events_rx,
            cancel: cancel_tx,
            task,
        })
    }
}

fn drive_stream(
    mut session: Session,
    settings: &NemotronConfig,
    mut samples_rx: mpsc::Receiver<Vec<f32>>,
    mut cancel_rx: oneshot::Receiver<()>,
    events_tx: &mpsc::Sender<StreamingEvent>,
) -> Result<(), TranscribeError> {
    let options = StreamOptions {
        family: Some(StreamExtension::ParakeetStream(ParakeetStreamOptions {
            att_context_right: Some(
                settings
                    .attention_right_context()
                    .map_err(TranscribeError::ConfigError)?,
            ),
        })),
        ..Default::default()
    };
    let mut stream = session
        .stream(&run_options(), &options)
        .map_err(native_error)?;
    let runtime = tokio::runtime::Handle::current();
    let chunk_samples = settings.streaming_chunk_ms as usize * 16;
    let mut pending = Vec::with_capacity(chunk_samples * 2);
    let mut last_preview = None;
    let mut cancelled = false;
    let mut cancellation_open = true;
    loop {
        let samples = runtime.block_on(async {
            tokio::select! {
                biased;
                signal = &mut cancel_rx, if cancellation_open => {
                    if signal.is_ok() {
                        cancelled = true;
                        None
                    } else {
                        cancellation_open = false;
                        Some(Vec::new())
                    }
                }
                samples = samples_rx.recv() => samples,
            }
        });
        let Some(samples) = samples else { break };
        validate_samples(&samples)?;
        pending.extend_from_slice(&samples);
        while pending.len() >= chunk_samples {
            let update = stream
                .feed(&pending[..chunk_samples])
                .map_err(native_error)?;
            pending.drain(..chunk_samples);
            if update.result_changed {
                let text = stream.text().display();
                let snapshot = stream.snapshot();
                let words = if snapshot.text.trim() == text.trim() {
                    word_confidences(&snapshot)
                } else {
                    Vec::new()
                };
                let preview = (text, words);
                if last_preview.as_ref() != Some(&preview) {
                    if events_tx
                        .blocking_send(preview_event(preview.0.clone(), preview.1.clone()))
                        .is_err()
                    {
                        return Ok(());
                    }
                    last_preview = Some(preview);
                }
            }
        }
    }
    if cancelled {
        return Ok(());
    }
    if !pending.is_empty() {
        stream.feed(&pending).map_err(native_error)?;
    }
    stream.finalize().map_err(native_error)?;
    let snapshot = stream.snapshot();
    let text = stream.text().full.trim().to_string();
    if !text.is_empty() {
        let words = if snapshot.text.trim() == text {
            word_confidences(&snapshot)
        } else {
            Vec::new()
        };
        let _ = events_tx.blocking_send(preview_event(text.clone(), words));
        let _ = events_tx.blocking_send(StreamingEvent::Final {
            text,
            segment_id: 0,
        });
    }
    Ok(())
}

fn preview_event(text: String, words: Vec<WordConfidence>) -> StreamingEvent {
    StreamingEvent::Preview {
        text,
        segment_id: 0,
        words,
    }
}

fn word_confidences(snapshot: &Transcript) -> Vec<WordConfidence> {
    let mut words = Vec::new();
    let mut text = String::new();
    let mut sum = 0.0;
    let mut count = 0;
    let finish_word =
        |words: &mut Vec<WordConfidence>, text: &mut String, sum: f32, count: usize| {
            if !text.is_empty() {
                words.push(WordConfidence {
                    text: std::mem::take(text),
                    confidence: (count > 0).then(|| sum / count as f32),
                });
            }
        };
    for token in &snapshot.tokens {
        let score = token.p.is_finite().then(|| token.p.clamp(0.0, 1.0));
        let mut token_in_word = false;
        for character in token.text.chars() {
            if character.is_whitespace() {
                finish_word(&mut words, &mut text, sum, count);
                sum = 0.0;
                count = 0;
                token_in_word = false;
            } else {
                text.push(character);
                if !token_in_word {
                    if let Some(score) = score {
                        sum += score;
                        count += 1;
                    }
                    token_in_word = true;
                }
            }
        }
    }
    finish_word(&mut words, &mut text, sum, count);
    if words
        .iter()
        .map(|word| word.text.as_str())
        .collect::<Vec<_>>()
        != snapshot.text.split_whitespace().collect::<Vec<_>>()
    {
        return Vec::new();
    }
    words
}

fn run_options() -> RunOptions {
    RunOptions {
        language: Some("en".to_string()),
        timestamps: TimestampKind::Token,
        ..Default::default()
    }
}

fn native_error(error: transcribe_cpp::Error) -> TranscribeError {
    TranscribeError::InferenceFailed(format!("Native Nemotron: {error}"))
}

fn validate_samples(samples: &[f32]) -> Result<(), TranscribeError> {
    if samples
        .iter()
        .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
    {
        return Err(TranscribeError::AudioFormat(
            "Nemotron requires finite 16 kHz mono PCM samples in [-1, 1]".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcribe_cpp::StreamText;

    #[tokio::test]
    #[ignore = "requires a local Nemotron GGUF model and CUDA runtime"]
    async fn nemotron_cuda_streaming_smoke() {
        let settings = NemotronConfig {
            model: std::env::var("VOXTYPE_NEMOTRON_MODEL")
                .expect("set VOXTYPE_NEMOTRON_MODEL to the GGUF path"),
            backend: "cuda".to_string(),
            ..Default::default()
        };
        let backend = NemotronTranscriber::new(&settings).unwrap();
        let mut wav = hound::WavReader::open("tests/fixtures/vad/speech_hello.wav").unwrap();
        assert_eq!(wav.spec().sample_rate, 16000);
        assert_eq!(wav.spec().channels, 1);
        let mut samples: Vec<f32> = wav
            .samples::<i16>()
            .map(|sample| sample.unwrap() as f32 / 32768.0)
            .collect();
        samples.extend(vec![0.0; 16000]);
        assert!(samples.len() < 16000 * 10);
        let mut long_wav = hound::WavReader::open("tests/fixtures/vad/speech_long.wav").unwrap();
        assert_eq!(long_wav.spec().sample_rate, 16000);
        assert_eq!(long_wav.spec().channels, 1);
        let continuous_speech: Vec<f32> = long_wav
            .samples::<i16>()
            .map(|sample| sample.unwrap() as f32 / 32768.0)
            .collect();
        assert!(!continuous_speech.is_empty());
        let mut long_samples = samples.clone();
        while long_samples.len() + samples.len() < 16000 * 20 {
            long_samples.extend_from_slice(&continuous_speech);
        }
        long_samples.extend_from_slice(&samples);
        for (audio, cancel, expected_hellos) in [
            (samples.clone(), false, 1),
            (samples.clone(), true, 0),
            (samples, false, 1),
            (long_samples, false, 2),
        ] {
            let (sender, capture_rx) = mpsc::channel(1);
            let (stream_tx, receiver) = mpsc::channel(1);
            let forwarder = crate::audio::levels::spawn_lossless_streaming_tap(
                capture_rx, None, stream_tx, None,
            );
            let StreamHandle {
                mut events,
                cancel: cancel_sender,
                task,
            } = backend.start_stream(receiver).unwrap();
            for chunk in audio.chunks(2560) {
                sender.send(chunk.to_vec()).await.unwrap();
            }
            let preview = tokio::time::timeout(std::time::Duration::from_secs(30), events.recv())
                .await
                .unwrap()
                .unwrap();
            assert!(
                matches!(preview, StreamingEvent::Preview { text, words, .. }
                if !text.trim().is_empty() && !words.is_empty()
                    && words.iter().any(|word| word.confidence.is_some()))
            );
            if cancel {
                cancel_sender.send(()).unwrap();
            } else {
                drop(cancel_sender);
            }
            drop(sender);
            let mut finals = Vec::new();
            let mut ended = 0;
            while let Some(event) =
                tokio::time::timeout(std::time::Duration::from_secs(30), events.recv())
                    .await
                    .unwrap()
            {
                match event {
                    StreamingEvent::Preview { .. } => {}
                    StreamingEvent::Final { text, .. } => finals.push(text),
                    StreamingEvent::Ended => ended += 1,
                    other => panic!("unexpected event: {other:?}"),
                }
            }
            task.await.unwrap().unwrap();
            forwarder.await.unwrap();
            assert_eq!(ended, 1);
            if cancel {
                assert!(finals.is_empty());
            } else {
                assert_eq!(finals.len(), 1);
                assert!(
                    finals[0].to_lowercase().matches("hello").count() >= expected_hellos,
                    "{}",
                    finals[0]
                );
            }
        }
    }

    #[test]
    fn nemotron_preview_is_not_a_typing_event() {
        let snapshot = StreamText {
            full: "raw revision".to_string(),
            committed: "Hello ".to_string(),
            tentative: "world".to_string(),
        };
        assert!(matches!(
            preview_event(snapshot.display(), Vec::new()),
            StreamingEvent::Preview { text, segment_id: 0, .. } if text == "Hello world"
        ));
    }

    #[test]
    fn nemotron_confidence_groups_subwords_and_ignores_missing_scores() {
        let snapshot = Transcript {
            text: "Hello world unknown".to_string(),
            tokens: vec![
                transcribe_cpp::Token {
                    text: " Hel".to_string(),
                    p: 0.9,
                    ..Default::default()
                },
                transcribe_cpp::Token {
                    text: "lo".to_string(),
                    p: 0.7,
                    ..Default::default()
                },
                transcribe_cpp::Token {
                    text: " world".to_string(),
                    p: 0.6,
                    ..Default::default()
                },
                transcribe_cpp::Token {
                    text: " unknown".to_string(),
                    p: f32::NAN,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let words = word_confidences(&snapshot);
        assert_eq!(
            words
                .iter()
                .map(|word| word.text.as_str())
                .collect::<Vec<_>>(),
            ["Hello", "world", "unknown"]
        );
        assert!((words[0].confidence.unwrap() - 0.8).abs() < 1e-6);
        assert_eq!(words[1].confidence, Some(0.6));
        assert_eq!(words[2].confidence, None);
        let revised = Transcript {
            text: "Different hypothesis".to_string(),
            ..snapshot
        };
        assert!(word_confidences(&revised).is_empty());
        assert!(word_confidences(&Transcript::default()).is_empty());
    }

    #[test]
    fn nemotron_rejects_invalid_pcm() {
        assert!(validate_samples(&[]).is_ok());
        assert!(validate_samples(&[-1.0, 0.0, 1.0]).is_ok());
        assert!(validate_samples(&[f32::NAN]).is_err());
        assert!(validate_samples(&[1.1]).is_err());
    }
}
