use super::granite::resolve_model_path;
use super::{StreamHandle, StreamingEvent, StreamingTranscriber, Transcriber, WordConfidence};
use crate::config::{LexiconConfig, NemotronConfig};
use crate::error::TranscribeError;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use transcribe_cpp::{
    Backend, CommitPolicy, Model, ModelOptions, ParakeetStreamOptions, RunOptions, Session,
    StreamExtension, StreamOptions, TimestampKind, Transcript,
};

pub struct NemotronTranscriber {
    model: Model,
    batch: Mutex<Session>,
    settings: NemotronConfig,
    lexicon: Option<Arc<Mutex<super::lexicon::Corrector>>>,
}

impl NemotronTranscriber {
    pub fn new(settings: &NemotronConfig) -> Result<Self, TranscribeError> {
        Self::new_with_lexicon(settings, &LexiconConfig::default())
    }

    pub fn new_with_lexicon(
        settings: &NemotronConfig,
        lexicon: &LexiconConfig,
    ) -> Result<Self, TranscribeError> {
        let lexicon = super::lexicon::Corrector::open(lexicon)?
            .map(|corrector| Arc::new(Mutex::new(corrector)));
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
            lexicon,
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
            .map(|transcript| {
                correct_text(
                    self.lexicon.as_ref(),
                    transcript.text.trim(),
                    &word_confidences(&transcript),
                )
            })
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
        let lexicon = self.lexicon.clone();
        let (events_tx, events_rx) = mpsc::channel(64);
        let (cancel_tx, cancel_rx) = oneshot::channel();
        let worker = tokio::task::spawn_blocking(move || {
            let outcome = drive_stream(
                session,
                &settings,
                lexicon.as_ref(),
                samples_rx,
                cancel_rx,
                &events_tx,
            );
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
    lexicon: Option<&Arc<Mutex<super::lexicon::Corrector>>>,
    mut samples_rx: mpsc::Receiver<Vec<f32>>,
    mut cancel_rx: oneshot::Receiver<()>,
    events_tx: &mpsc::Sender<StreamingEvent>,
) -> Result<(), TranscribeError> {
    use voxtype_lexicon::stream::{CommitBuffer, CommitMode};
    if let Some(mut corrector) = lexicon.and_then(|corrector| corrector.lock().ok()) {
        corrector.refresh();
    }
    let mode = lexicon
        .and_then(|corrector| corrector.lock().ok())
        .map(|corrector| corrector.commit_mode());
    let mut commits = mode
        .filter(|mode| *mode != CommitMode::Eou)
        .map(CommitBuffer::new);
    let mut committed_display = String::new();
    let mut committed_words = Vec::new();
    let options = StreamOptions {
        commit_policy: if commits.is_some() {
            CommitPolicy::StablePrefix
        } else {
            CommitPolicy::Auto
        },
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
            if update.committed_changed {
                if let Some(buffer) = &mut commits {
                    let stable = stream.text().committed;
                    let snapshot = stream.snapshot();
                    if let Some((text, words)) = commit_stable(
                        buffer,
                        mode.unwrap_or(CommitMode::Off),
                        lexicon,
                        &stable,
                        &snapshot,
                        false,
                    ) {
                        if !committed_display.is_empty() {
                            committed_display.push(' ');
                        }
                        committed_display.push_str(&text);
                        committed_words.extend(words);
                        if events_tx
                            .blocking_send(StreamingEvent::Final {
                                text,
                                segment_id: 0,
                            })
                            .is_err()
                        {
                            return Ok(());
                        }
                    }
                }
            }
            if update.result_changed {
                let text = stream.text().display();
                let snapshot = stream.snapshot();
                let words = if snapshot.text.trim() == text.trim() {
                    word_confidences(&snapshot)
                } else {
                    Vec::new()
                };
                let preview = if commits.is_none() {
                    lexicon_preview(lexicon, text, words)
                } else {
                    streaming_preview(commits.as_ref(), &committed_words, text, words)
                };
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
        let (text, words) = if let Some(buffer) = &mut commits {
            if let Some((tail, words)) = commit_stable(
                buffer,
                mode.unwrap_or(CommitMode::Off),
                lexicon,
                &text,
                &snapshot,
                true,
            ) {
                if !committed_display.is_empty() {
                    committed_display.push(' ');
                }
                committed_display.push_str(&tail);
                committed_words.extend(words);
                let _ = events_tx.blocking_send(StreamingEvent::Final {
                    text: tail,
                    segment_id: 0,
                });
            }
            (committed_display, committed_words)
        } else {
            correct_scored(lexicon, &text, &words)
        };
        let _ = events_tx.blocking_send(preview_event(text.clone(), words));
        if commits.is_none() {
            let _ = events_tx.blocking_send(StreamingEvent::Final {
                text,
                segment_id: 0,
            });
        }
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

fn correct_text(
    lexicon: Option<&Arc<Mutex<super::lexicon::Corrector>>>,
    text: &str,
    words: &[WordConfidence],
) -> String {
    correct_scored(lexicon, text, words).0
}

fn correct_scored(
    lexicon: Option<&Arc<Mutex<super::lexicon::Corrector>>>,
    text: &str,
    words: &[WordConfidence],
) -> (String, Vec<WordConfidence>) {
    match lexicon.and_then(|lexicon| lexicon.lock().ok()) {
        Some(mut corrector) => corrector.correct(text, words),
        None => (text.to_string(), words.to_vec()),
    }
}

fn streaming_preview(
    buffer: Option<&voxtype_lexicon::stream::CommitBuffer>,
    committed: &[WordConfidence],
    text: String,
    words: Vec<WordConfidence>,
) -> (String, Vec<WordConfidence>) {
    if committed.is_empty() {
        return (text, words);
    }
    let raw: Vec<_> = text
        .split_whitespace()
        .map(|text| voxtype_lexicon::query::Word {
            text,
            confidence: None,
        })
        .collect();
    let Some(pending) = buffer.and_then(|buffer| buffer.pending(&raw)) else {
        return (text, words);
    };
    let mut preview = committed.to_vec();
    let offset = raw.len() - pending.len();
    let scores_match = words
        .iter()
        .map(|word| word.text.as_str())
        .eq(raw.iter().map(|word| word.text));
    if scores_match {
        preview.extend_from_slice(&words[offset..]);
    } else {
        preview.extend(pending.iter().map(|word| WordConfidence {
            text: word.text.into(),
            confidence: None,
            lexicon: false,
        }));
    }
    let text = preview
        .iter()
        .map(|word| word.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    (text, preview)
}

fn lexicon_preview(
    lexicon: Option<&Arc<Mutex<super::lexicon::Corrector>>>,
    text: String,
    words: Vec<WordConfidence>,
) -> (String, Vec<WordConfidence>) {
    if words.is_empty() {
        return (text, words);
    }
    let Some(mut corrector) = lexicon.and_then(|corrector| corrector.try_lock().ok()) else {
        return (text, words);
    };
    let query_words: Vec<_> = words
        .iter()
        .map(|word| voxtype_lexicon::query::Word {
            text: &word.text,
            confidence: word.confidence,
        })
        .collect();
    let result = corrector.preview_query(&query_words);
    if result.replacements.is_empty() {
        (text, words)
    } else {
        (
            result.text,
            super::lexicon::replacement_words(&words, &result.replacements),
        )
    }
}

fn commit_stable(
    buffer: &mut voxtype_lexicon::stream::CommitBuffer,
    mode: voxtype_lexicon::stream::CommitMode,
    lexicon: Option<&Arc<Mutex<super::lexicon::Corrector>>>,
    stable: &str,
    snapshot: &Transcript,
    finalizing: bool,
) -> Option<(String, Vec<WordConfidence>)> {
    use voxtype_lexicon::query::{QueryResult, Word};
    let mut raw_words: Vec<_> = stable.split_whitespace().collect();
    if !finalizing && !stable.ends_with(char::is_whitespace) {
        raw_words.pop();
    }
    let scored = word_confidences(snapshot);
    let scores_match = scored
        .iter()
        .map(|word| word.text.as_str())
        .take(raw_words.len())
        .eq(raw_words.iter().copied());
    let words: Vec<_> = raw_words
        .iter()
        .enumerate()
        .map(|(position, text)| Word {
            text,
            confidence: scores_match
                .then(|| scored.get(position).and_then(|word| word.confidence))
                .flatten(),
        })
        .collect();
    let Some(pending) = buffer.pending(&words) else {
        tracing::warn!("Nemotron revised a typed stable prefix; leaving typed text unchanged");
        return None;
    };
    let result = if mode != voxtype_lexicon::stream::CommitMode::Off {
        lexicon
            .and_then(|corrector| corrector.try_lock().ok())
            .map(|mut corrector| corrector.stream_query(pending))
    } else {
        None
    };
    let result = result.unwrap_or_else(|| QueryResult {
        text: String::new(),
        lookup_words: Vec::new(),
        replacements: Vec::new(),
        ambiguous: Vec::new(),
    });
    let pending_count = pending.len();
    let offset = words.len() - pending_count;
    let text = buffer.commit(&words, &result, finalizing)?;
    let count = pending_count - buffer.pending(&words)?.len();
    let source: Vec<_> = words[offset..offset + count]
        .iter()
        .map(|word| WordConfidence {
            text: word.text.into(),
            confidence: word.confidence,
            lexicon: false,
        })
        .collect();
    Some((
        text,
        super::lexicon::replacement_words(&source, &result.replacements),
    ))
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
                    lexicon: false,
                });
            }
        };
    for token in &snapshot.tokens {
        let score = (token.p.is_finite() && (0.0..=1.0).contains(&token.p)).then_some(token.p);
        if token.p.is_finite() && score.is_none() {
            tracing::warn!(token = %token.text, score = token.p, "Invalid Nemotron confidence: expected probability in 0..1, not percent");
        }
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

    #[test]
    fn streaming_preview_keeps_lexicon_picks_in_committed_prefix() {
        use voxtype_lexicon::query::{QueryResult, Replacement, Word};
        use voxtype_lexicon::stream::{CommitBuffer, CommitMode};
        let source: Vec<_> = ["cube", "control", "apply", "now", "please"]
            .into_iter()
            .map(|text| Word {
                text,
                confidence: Some(0.88),
            })
            .collect();
        let result = QueryResult {
            text: "kubectl apply now please".into(),
            lookup_words: Vec::new(),
            replacements: vec![Replacement {
                start: 0,
                end: 2,
                canonical: "kubectl".into(),
                score: 1.0,
                gain: 0.12,
            }],
            ambiguous: Vec::new(),
        };
        let mut buffer = CommitBuffer::new(CommitMode::Lag);
        assert_eq!(
            buffer.commit(&source, &result, false).as_deref(),
            Some("kubectl")
        );
        let original: Vec<_> = source
            .iter()
            .map(|word| WordConfidence {
                text: word.text.into(),
                confidence: word.confidence,
                lexicon: false,
            })
            .collect();
        let committed =
            super::super::lexicon::replacement_words(&original[..2], &result.replacements);
        for scores in [original.clone(), Vec::new()] {
            let (text, words) = streaming_preview(
                Some(&buffer),
                &committed,
                "cube control apply now please".into(),
                scores,
            );
            assert_eq!(text, "kubectl apply now please");
            assert_eq!(words[0], committed[0]);
            assert!(words[0].lexicon);
            assert!(words[1..].iter().all(|word| !word.lexicon));
        }
    }

    #[test]
    fn stable_prefix_commits_only_untyped_complete_words() {
        use voxtype_lexicon::stream::{CommitBuffer, CommitMode};
        let snapshot = Transcript {
            text: "one two three four five".into(),
            ..Default::default()
        };
        let mut buffer = CommitBuffer::new(CommitMode::Lag);
        assert_eq!(
            commit_stable(
                &mut buffer,
                CommitMode::Off,
                None,
                "one two three four five ",
                &snapshot,
                false
            )
            .as_ref()
            .map(|(text, _)| text.as_str()),
            Some("one two")
        );
        assert_eq!(
            commit_stable(
                &mut buffer,
                CommitMode::Off,
                None,
                "one two three four five",
                &snapshot,
                true
            )
            .as_ref()
            .map(|(text, _)| text.as_str()),
            Some("three four five")
        );
        assert!(commit_stable(
            &mut buffer,
            CommitMode::Off,
            None,
            "one two three four five",
            &snapshot,
            true
        )
        .is_none());
        let mut immediate = CommitBuffer::new(CommitMode::Off);
        assert_eq!(
            commit_stable(
                &mut immediate,
                CommitMode::Off,
                None,
                "one tw",
                &snapshot,
                false
            )
            .as_ref()
            .map(|(text, _)| text.as_str()),
            Some("one")
        );
    }

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
    fn nemotron_confidence_uses_probability_not_percentage() {
        for (probability, expected) in [
            (0.88, Some(0.88)),
            (0.89, Some(0.89)),
            (88.0, None),
            (-0.1, None),
        ] {
            let snapshot = Transcript {
                text: "word".into(),
                tokens: vec![transcribe_cpp::Token {
                    text: " word".into(),
                    p: probability,
                    ..Default::default()
                }],
                ..Default::default()
            };
            let words = word_confidences(&snapshot);
            assert_eq!(words[0].confidence, expected);
            if let Some(confidence) = words[0].confidence {
                assert!((confidence * 100.0 - probability * 100.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn nemotron_rejects_invalid_pcm() {
        assert!(validate_samples(&[]).is_ok());
        assert!(validate_samples(&[-1.0, 0.0, 1.0]).is_ok());
        assert!(validate_samples(&[f32::NAN]).is_err());
        assert!(validate_samples(&[1.1]).is_err());
    }
}
