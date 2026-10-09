use crate::transcribe::WordConfidence;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct TranscriptPopupConfig {
    pub enabled: bool,
    pub review_mode: bool,
    pub font_size: f32,
    pub opacity: f32,
    pub width_px: u32,
    pub height_px: u32,
    pub final_display_ms: u64,
}

impl Default for TranscriptPopupConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            review_mode: false,
            font_size: 28.0,
            opacity: 0.8,
            width_px: 760,
            height_px: 280,
            final_display_ms: 2000,
        }
    }
}

impl TranscriptPopupConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.review_mode && !self.enabled {
            return Err(
                "transcript_popup.review_mode requires transcript_popup.enabled = true".to_string(),
            );
        }
        if self.review_mode && self.height_px < 160 {
            return Err(
                "transcript_popup.review_mode requires height_px >= 160 for review controls"
                    .to_string(),
            );
        }
        if !self.font_size.is_finite() || !(12.0..=72.0).contains(&self.font_size) {
            return Err("transcript_popup.font_size must be between 12 and 72".to_string());
        }
        if !self.opacity.is_finite() || !(0.0..=1.0).contains(&self.opacity) {
            return Err("transcript_popup.opacity must be between 0 and 1".to_string());
        }
        if !(240..=1920).contains(&self.width_px) || !(80..=1080).contains(&self.height_px) {
            return Err(
                "transcript_popup dimensions must be 240..1920 by 80..1080 pixels".to_string(),
            );
        }
        if self.final_display_ms > 60_000 {
            return Err("transcript_popup.final_display_ms must not exceed 60000".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TranscriptSnapshot {
    pub text: String,
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub pending_delivery: bool,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub words: Vec<WordConfidence>,
    pub recording: bool,
    pub expires_at_ms: u64,
    pub settings: TranscriptPopupConfig,
}

impl TranscriptSnapshot {
    pub fn visible_at(&self, now_ms: u64) -> bool {
        self.settings.enabled
            && (self.settings.review_mode || self.recording || now_ms < self.expires_at_ms)
    }

    pub fn read(path: &Path) -> Option<Self> {
        serde_json::from_slice(&std::fs::read(path).ok()?).ok()
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ReviewAction {
    Deliver,
    Discard,
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct ReviewRequest {
    pub revision: u64,
    pub action: ReviewAction,
}

pub fn request_review_action(
    path: &Path,
    revision: u64,
    action: ReviewAction,
) -> std::io::Result<()> {
    let snapshot = TranscriptSnapshot::read(path).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Transcript is no longer available",
        )
    })?;
    if !snapshot.settings.review_mode || !snapshot.pending_delivery || snapshot.revision != revision
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Transcript changed; review the current text",
        ));
    }
    let directory = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Popup path has no directory",
        )
    })?;
    let mut temporary = tempfile::NamedTempFile::new_in(directory)?;
    serde_json::to_writer(&mut temporary, &ReviewRequest { revision, action })?;
    temporary.flush()?;
    temporary
        .persist(path.with_extension("action.json"))
        .map_err(|error| error.error)?;
    Ok(())
}

pub struct TranscriptPublisher {
    path: PathBuf,
    settings: TranscriptPopupConfig,
    revision: std::sync::Arc<std::sync::Mutex<u64>>,
}

impl TranscriptPublisher {
    pub fn new(path: PathBuf, settings: TranscriptPopupConfig) -> std::io::Result<Self> {
        std::fs::create_dir_all(path.parent().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Popup path has no directory",
            )
        })?)?;
        Ok(Self {
            path,
            settings,
            revision: Default::default(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn publish(&self, text: &str, recording: bool) -> std::io::Result<()> {
        if !text.trim().is_empty() {
            if let Some(snapshot) = TranscriptSnapshot::read(&self.path) {
                if snapshot.text.trim() == text.trim() {
                    return self
                        .publish_snapshot(
                            text,
                            recording,
                            &snapshot.words,
                            false,
                            snapshot.confidence,
                        )
                        .map(|_| ());
                }
                if !recording {
                    return self
                        .publish_snapshot(text, recording, &[], false, snapshot.confidence)
                        .map(|_| ());
                }
            }
        }
        self.publish_scored(text, recording, &[])
    }

    pub fn publish_scored(
        &self,
        text: &str,
        recording: bool,
        words: &[WordConfidence],
    ) -> std::io::Result<()> {
        if words.is_empty() && !text.trim().is_empty() {
            if let Some(snapshot) = TranscriptSnapshot::read(&self.path) {
                if snapshot.text.trim() == text.trim() {
                    return self
                        .publish_snapshot(
                            text,
                            recording,
                            &snapshot.words,
                            false,
                            snapshot.confidence,
                        )
                        .map(|_| ());
                }
            }
        }
        self.publish_snapshot(text, recording, words, false, None)
            .map(|_| ())
    }

    pub fn stage_for_review(&self, text: &str) -> std::io::Result<u64> {
        let snapshot = TranscriptSnapshot::read(&self.path);
        let confidence = snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.confidence)
            .filter(|_| !text.trim().is_empty());
        let words = snapshot
            .filter(|snapshot| snapshot.text.trim() == text.trim())
            .map(|snapshot| snapshot.words)
            .unwrap_or_default();
        self.publish_snapshot(text, false, &words, !text.trim().is_empty(), confidence)
    }

    pub fn take_review_request(&self) -> std::io::Result<Option<ReviewRequest>> {
        let path = self.path.with_extension("action.json");
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        std::fs::remove_file(path)?;
        let request: ReviewRequest = serde_json::from_slice(&bytes)?;
        let current = TranscriptSnapshot::read(&self.path);
        Ok(current
            .filter(|snapshot| {
                snapshot.settings.review_mode
                    && snapshot.pending_delivery
                    && snapshot.revision == request.revision
            })
            .map(|_| request))
    }

    fn publish_snapshot(
        &self,
        text: &str,
        recording: bool,
        words: &[WordConfidence],
        pending_delivery: bool,
        confidence_hint: Option<f32>,
    ) -> std::io::Result<u64> {
        if !self.settings.enabled {
            return Ok(0);
        }
        let words: Vec<_> = words
            .iter()
            .map(|word| WordConfidence {
                text: word.text.clone(),
                confidence: word
                    .confidence
                    .filter(|score| score.is_finite() && (0.0..=1.0).contains(score)),
            })
            .collect();
        let scores: Vec<_> = words.iter().filter_map(|word| word.confidence).collect();
        let confidence = (!scores.is_empty())
            .then(|| scores.iter().sum::<f32>() / scores.len() as f32)
            .or_else(|| {
                confidence_hint.filter(|score| score.is_finite() && (0.0..=1.0).contains(score))
            });
        let mut revision = self.revision.lock().expect("popup revision poisoned");
        *revision += 1;
        let published_revision = *revision;
        let snapshot = TranscriptSnapshot {
            text: text.to_string(),
            revision: published_revision,
            pending_delivery,
            confidence,
            words,
            recording,
            expires_at_ms: now_ms().saturating_add(self.settings.final_display_ms),
            settings: self.settings.clone(),
        };
        let directory = self.path.parent().expect("validated popup directory");
        let mut temporary = tempfile::NamedTempFile::new_in(directory)?;
        serde_json::to_writer(&mut temporary, &snapshot)?;
        temporary.flush()?;
        temporary.persist(&self.path).map_err(|error| error.error)?;
        if !recording && !self.settings.review_mode {
            if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                let path = self.path.clone();
                let revision = self.revision.clone();
                let delay = self.settings.final_display_ms;
                runtime.spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                    let current = revision.lock().expect("popup revision poisoned");
                    if *current == published_revision {
                        let _ = std::fs::remove_file(path);
                    }
                });
            }
        }
        Ok(published_revision)
    }

    pub fn clear(&self) {
        let mut revision = self.revision.lock().expect("popup revision poisoned");
        *revision += 1;
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_file(self.path.with_extension("action.json"));
    }
}

impl Drop for TranscriptPublisher {
    fn drop(&mut self) {
        self.clear();
    }
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    unsafe fn click_review_window(geometry: &str, horizontal: i32) {
        type Display = *mut libc::c_void;
        unsafe fn symbol<Function: Copy>(
            handle: *mut libc::c_void,
            name: &std::ffi::CStr,
        ) -> Function {
            let pointer = libc::dlsym(handle, name.as_ptr());
            assert!(!pointer.is_null(), "missing X11 test function: {name:?}");
            std::mem::transmute_copy(&pointer)
        }
        let xlib = libc::dlopen(c"libX11.so.6".as_ptr(), libc::RTLD_NOW);
        let xtest = libc::dlopen(c"libXtst.so.6".as_ptr(), libc::RTLD_NOW);
        assert!(
            !xlib.is_null() && !xtest.is_null(),
            "graphical click test requires X11 and XTest libraries"
        );
        let open: unsafe extern "C" fn(*const libc::c_char) -> Display =
            symbol(xlib, c"XOpenDisplay");
        let close: unsafe extern "C" fn(Display) -> libc::c_int = symbol(xlib, c"XCloseDisplay");
        let sync: unsafe extern "C" fn(Display, libc::c_int) -> libc::c_int =
            symbol(xlib, c"XSync");
        let root: unsafe extern "C" fn(Display) -> libc::c_ulong =
            symbol(xlib, c"XDefaultRootWindow");
        let query: unsafe extern "C" fn(
            Display,
            libc::c_ulong,
            *mut libc::c_ulong,
            *mut libc::c_ulong,
            *mut libc::c_int,
            *mut libc::c_int,
            *mut libc::c_int,
            *mut libc::c_int,
            *mut libc::c_uint,
        ) -> libc::c_int = symbol(xlib, c"XQueryPointer");
        let motion: unsafe extern "C" fn(
            Display,
            libc::c_int,
            libc::c_int,
            libc::c_int,
            libc::c_ulong,
        ) -> libc::c_int = symbol(xtest, c"XTestFakeMotionEvent");
        let button: unsafe extern "C" fn(
            Display,
            libc::c_uint,
            libc::c_int,
            libc::c_ulong,
        ) -> libc::c_int = symbol(xtest, c"XTestFakeButtonEvent");
        let display = open(std::ptr::null());
        assert!(!display.is_null());
        let (
            mut root_return,
            mut child_return,
            mut original_x,
            mut original_y,
            mut local_x,
            mut local_y,
            mut mask,
        ) = (0, 0, 0, 0, 0, 0, 0);
        assert_ne!(
            query(
                display,
                root(display),
                &mut root_return,
                &mut child_return,
                &mut original_x,
                &mut original_y,
                &mut local_x,
                &mut local_y,
                &mut mask
            ),
            0
        );
        let coordinate = |label: &str| {
            geometry
                .lines()
                .find_map(|line| {
                    line.trim()
                        .strip_prefix(label)
                        .and_then(|value| value.trim().parse::<i32>().ok())
                })
                .unwrap()
        };
        motion(
            display,
            -1,
            coordinate("Absolute upper-left X:") + horizontal,
            coordinate("Absolute upper-left Y:") + 240,
            0,
        );
        sync(display, 0);
        std::thread::sleep(std::time::Duration::from_millis(75));
        button(display, 1, 1, 0);
        sync(display, 0);
        std::thread::sleep(std::time::Duration::from_millis(75));
        button(display, 1, 0, 0);
        sync(display, 0);
        std::thread::sleep(std::time::Duration::from_millis(75));
        motion(display, -1, original_x, original_y, 0);
        sync(display, 0);
        close(display);
        libc::dlclose(xtest);
        libc::dlclose(xlib);
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    #[ignore = "requires XWayland, XTest, and a built native popup binary"]
    async fn transcript_popup_review_native_smoke() {
        struct ChildGuard(std::process::Child);
        impl Drop for ChildGuard {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        let focus = || {
            std::process::Command::new("xprop")
                .args(["-root", "_NET_ACTIVE_WINDOW"])
                .output()
                .unwrap()
                .stdout
        };
        let before = focus();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("transcript.json");
        let publisher = TranscriptPublisher::new(
            path.clone(),
            TranscriptPopupConfig {
                enabled: true,
                review_mode: true,
                final_display_ms: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let text = "Hello world. This is the text from the quiet room by the window.";
        let words: Vec<_> = text
            .split_whitespace()
            .map(|word| WordConfidence {
                text: word.to_string(),
                confidence: Some(if word == "quiet" { 0.54 } else { 0.92 }),
            })
            .collect();
        publisher.publish_scored(text, true, &words).unwrap();
        let mut child = ChildGuard(
            std::process::Command::new("target/nemotron-cuda/debug/voxtype-osd-native")
                .arg("--transcript-state")
                .arg(&path)
                .spawn()
                .unwrap(),
        );
        let window = tokio::time::timeout(std::time::Duration::from_secs(15), async {
            loop {
                assert!(child.0.try_wait().unwrap().is_none());
                let tree = std::process::Command::new("xwininfo")
                    .args(["-root", "-tree"])
                    .output()
                    .unwrap();
                for line in String::from_utf8_lossy(&tree.stdout)
                    .lines()
                    .filter(|line| line.contains("\"Voxtype Transcript\""))
                {
                    let Some(window) = line.split_whitespace().next() else {
                        continue;
                    };
                    let properties = std::process::Command::new("xprop")
                        .args(["-id", window, "_NET_WM_PID"])
                        .output()
                        .unwrap();
                    if String::from_utf8_lossy(&properties.stdout)
                        .split('=')
                        .nth(1)
                        .and_then(|pid| pid.trim().parse::<u32>().ok())
                        == Some(child.0.id())
                    {
                        return window.to_string();
                    }
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        })
        .await
        .unwrap();
        let geometry = std::process::Command::new("xwininfo")
            .args(["-id", &window])
            .output()
            .unwrap();
        let geometry = String::from_utf8(geometry.stdout).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        unsafe {
            click_review_window(&geometry, 72);
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert!(
            publisher.take_review_request().unwrap().is_none(),
            "Deliver was enabled while recording"
        );
        for (action, horizontal) in [(ReviewAction::Deliver, 72), (ReviewAction::Discard, 176)] {
            let revision = publisher.stage_for_review(text).unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            assert!(std::process::Command::new("xwd")
                .args([
                    "-id",
                    &window,
                    "-silent",
                    "-out",
                    "target/transcript-review-smoke.xwd"
                ])
                .status()
                .unwrap()
                .success());
            unsafe {
                click_review_window(&geometry, horizontal);
            }
            let request = tokio::time::timeout(std::time::Duration::from_secs(5), async {
                loop {
                    if let Some(request) = publisher.take_review_request().unwrap() {
                        return request;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            })
            .await
            .unwrap();
            assert_eq!(request, ReviewRequest { revision, action });
            assert_eq!(before, focus(), "review click changed the active window");
            publisher.publish("", false).unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            assert!(std::process::Command::new("xwininfo")
                .args(["-id", &window])
                .output()
                .unwrap()
                .status
                .success());
            assert!(TranscriptSnapshot::read(&path)
                .unwrap()
                .visible_at(u64::MAX));
        }
    }

    #[tokio::test]
    #[ignore = "requires a graphical session and a built native popup binary"]
    async fn transcript_popup_native_smoke() {
        struct ChildGuard(std::process::Child);
        impl Drop for ChildGuard {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        let focus = || {
            std::process::Command::new("xprop")
                .args(["-root", "_NET_ACTIVE_WINDOW"])
                .output()
                .unwrap()
                .stdout
        };
        let window = || {
            std::process::Command::new("xwininfo")
                .args(["-name", "Voxtype Transcript"])
                .output()
                .unwrap()
        };
        let before = focus();
        let directory = tempfile::tempdir().unwrap();
        let publisher = TranscriptPublisher::new(
            directory.path().join("transcript.json"),
            TranscriptPopupConfig {
                enabled: true,
                final_display_ms: 2000,
                ..Default::default()
            },
        )
        .unwrap();
        let text = "Hello world. This is a recording from the quiet room by the window.";
        let words: Vec<_> = text
            .split_whitespace()
            .map(|word| WordConfidence {
                text: word.to_string(),
                confidence: Some(if word == "quiet" { 0.54 } else { 0.92 }),
            })
            .collect();
        publisher.publish_scored(text, true, &words).unwrap();
        let mut child = ChildGuard(
            std::process::Command::new("target/nemotron-cuda/debug/voxtype-osd-native")
                .arg("--transcript-state")
                .arg(publisher.path())
                .spawn()
                .unwrap(),
        );
        tokio::time::timeout(std::time::Duration::from_secs(15), async {
            loop {
                assert!(
                    child.0.try_wait().unwrap().is_none(),
                    "popup frontend exited"
                );
                if window().status.success() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        })
        .await
        .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
        assert_eq!(before, focus(), "popup changed the active window");
        let info = window();
        let geometry = String::from_utf8(info.stdout).unwrap();
        assert!(geometry.contains("Width: 760"), "{geometry}");
        assert!(geometry.contains("Height: 280"), "{geometry}");
        assert!(std::process::Command::new("xwd")
            .args([
                "-name",
                "Voxtype Transcript",
                "-silent",
                "-out",
                "target/transcript-popup-smoke.xwd"
            ])
            .status()
            .unwrap()
            .success());
        publisher
            .publish(
                "Final transcript. This popup disappears after two seconds.",
                false,
            )
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
        assert!(window().status.success());
        tokio::time::sleep(std::time::Duration::from_millis(1250)).await;
        assert!(
            !window().status.success(),
            "popup remained visible after expiry"
        );
        assert_eq!(before, focus(), "closing popup changed the active window");
        assert!(!publisher.path().exists());
    }

    #[tokio::test]
    async fn transcript_popup_expiry_cleans_text_but_preserves_new_recording() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("transcript.json");
        let publisher = TranscriptPublisher::new(
            path.clone(),
            TranscriptPopupConfig {
                enabled: true,
                final_display_ms: 5,
                ..Default::default()
            },
        )
        .unwrap();
        publisher.publish("final", false).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        assert!(!path.exists());
        publisher.publish("old final", false).unwrap();
        publisher.publish("new recording", true).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        assert_eq!(
            TranscriptSnapshot::read(&path).unwrap().text,
            "new recording"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn transcript_popup_defaults_are_disabled_and_square_style() {
        let settings: TranscriptPopupConfig = toml::from_str("").unwrap();
        assert!(!settings.enabled);
        assert_eq!(settings.font_size, 28.0);
        assert_eq!(settings.opacity, 0.8);
        assert_eq!(settings.final_display_ms, 2000);
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn transcript_popup_final_text_expires_without_hiding_recording() {
        let mut snapshot = TranscriptSnapshot {
            text: "Hello".to_string(),
            revision: 0,
            pending_delivery: false,
            confidence: None,
            words: Vec::new(),
            recording: false,
            expires_at_ms: 3000,
            settings: TranscriptPopupConfig {
                enabled: true,
                ..Default::default()
            },
        };
        assert!(snapshot.visible_at(2999));
        assert!(!snapshot.visible_at(3000));
        snapshot.recording = true;
        assert!(snapshot.visible_at(4000));
        snapshot.settings.enabled = false;
        assert!(!snapshot.visible_at(4000));
    }

    #[test]
    fn transcript_popup_confidence_round_trips_and_clears() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("transcript.json");
        let publisher = TranscriptPublisher::new(
            path.clone(),
            TranscriptPopupConfig {
                enabled: true,
                ..Default::default()
            },
        )
        .unwrap();
        let words = vec![
            WordConfidence {
                text: "Hello".to_string(),
                confidence: Some(0.8),
            },
            WordConfidence {
                text: "world".to_string(),
                confidence: Some(0.6),
            },
            WordConfidence {
                text: "unknown".to_string(),
                confidence: Some(f32::NAN),
            },
        ];
        publisher
            .publish_scored("Hello world unknown", true, &words)
            .unwrap();
        let snapshot = TranscriptSnapshot::read(&path).unwrap();
        assert!((snapshot.confidence.unwrap() - 0.7).abs() < 1e-6);
        assert_eq!(snapshot.words[0], words[0]);
        assert_eq!(snapshot.words[1], words[1]);
        assert_eq!(snapshot.words[2].confidence, None);
        publisher.publish("Hello world unknown", false).unwrap();
        assert!(TranscriptSnapshot::read(&path)
            .unwrap()
            .confidence
            .is_some());
        publisher.publish("Changed final text", false).unwrap();
        assert!(TranscriptSnapshot::read(&path).unwrap().words.is_empty());
        publisher.publish("New recording", true).unwrap();
        let snapshot = TranscriptSnapshot::read(&path).unwrap();
        assert_eq!(snapshot.confidence, None);
        assert!(snapshot.words.is_empty());
    }

    #[test]
    fn transcript_popup_accepts_legacy_snapshots_without_confidence() {
        let snapshot: TranscriptSnapshot = serde_json::from_value(serde_json::json!({
            "text": "Hello", "recording": true, "expires_at_ms": 3000,
            "settings": { "enabled": true }
        }))
        .unwrap();
        assert_eq!(snapshot.confidence, None);
        assert!(snapshot.words.is_empty());
    }

    #[test]
    fn transcript_popup_disabled_writes_nothing_and_clear_removes_text() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("transcript.json");
        let publisher =
            TranscriptPublisher::new(path.clone(), TranscriptPopupConfig::default()).unwrap();
        publisher.publish("private words", true).unwrap();
        assert!(!path.exists());
        drop(publisher);
        let publisher = TranscriptPublisher::new(
            path.clone(),
            TranscriptPopupConfig {
                enabled: true,
                ..Default::default()
            },
        )
        .unwrap();
        publisher.publish("Hello", true).unwrap();
        assert_eq!(TranscriptSnapshot::read(&path).unwrap().text, "Hello");
        publisher.clear();
        assert!(!path.exists());
    }

    #[test]
    fn transcript_popup_preserves_scores_through_release_handoff() {
        for review_mode in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("transcript.json");
            let publisher = TranscriptPublisher::new(
                path.clone(),
                TranscriptPopupConfig {
                    enabled: true,
                    review_mode,
                    ..Default::default()
                },
            )
            .unwrap();
            let words = [WordConfidence {
                text: "hello".to_string(),
                confidence: Some(0.85),
            }];
            publisher.publish_scored("hello", true, &words).unwrap();
            publisher.publish_scored("hello", true, &[]).unwrap();
            publisher.publish("hello", review_mode).unwrap();
            if review_mode {
                publisher.stage_for_review("hello").unwrap();
            }
            let snapshot = TranscriptSnapshot::read(&path).unwrap();
            assert_eq!(snapshot.confidence, Some(0.85));
            assert_eq!(snapshot.words, words);
            assert!(!snapshot.recording);
            publisher.publish("", true).unwrap();
            let snapshot = TranscriptSnapshot::read(&path).unwrap();
            assert_eq!(snapshot.confidence, None);
            assert!(snapshot.words.is_empty());
        }
    }

    #[test]
    fn transcript_popup_review_retains_overall_confidence_after_text_processing() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("transcript.json");
        let publisher = TranscriptPublisher::new(
            path.clone(),
            TranscriptPopupConfig {
                enabled: true,
                review_mode: true,
                ..Default::default()
            },
        )
        .unwrap();
        publisher
            .publish_scored(
                "hello world",
                true,
                &[
                    WordConfidence {
                        text: "hello".to_string(),
                        confidence: Some(0.8),
                    },
                    WordConfidence {
                        text: "world".to_string(),
                        confidence: Some(0.6),
                    },
                ],
            )
            .unwrap();
        for _ in 0..2 {
            publisher.stage_for_review("Hello, world!").unwrap();
            let snapshot = TranscriptSnapshot::read(&path).unwrap();
            assert!((snapshot.confidence.unwrap() - 0.7).abs() < 1e-6);
            assert!(snapshot.words.is_empty());
            assert!(snapshot.pending_delivery);
        }
        publisher.publish("", false).unwrap();
        assert_eq!(TranscriptSnapshot::read(&path).unwrap().confidence, None);
    }

    #[test]
    fn transcript_popup_review_requests_are_revision_checked_and_consumed_once() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("transcript.json");
        let publisher = TranscriptPublisher::new(
            path.clone(),
            TranscriptPopupConfig {
                enabled: true,
                review_mode: true,
                ..Default::default()
            },
        )
        .unwrap();
        publisher.publish("", false).unwrap();
        assert!(TranscriptSnapshot::read(&path)
            .unwrap()
            .visible_at(u64::MAX));
        let revision = publisher.stage_for_review("Review these words").unwrap();
        request_review_action(&path, revision, ReviewAction::Deliver).unwrap();
        assert_eq!(
            publisher.take_review_request().unwrap(),
            Some(ReviewRequest {
                revision,
                action: ReviewAction::Deliver
            })
        );
        assert!(publisher.take_review_request().unwrap().is_none());
        request_review_action(&path, revision, ReviewAction::Deliver).unwrap();
        let next = publisher.stage_for_review("New words").unwrap();
        assert!(publisher.take_review_request().unwrap().is_none());
        assert!(request_review_action(&path, revision, ReviewAction::Deliver).is_err());
        request_review_action(&path, next, ReviewAction::Discard).unwrap();
        assert_eq!(
            publisher.take_review_request().unwrap().unwrap().action,
            ReviewAction::Discard
        );
        publisher.publish("", false).unwrap();
        assert!(!TranscriptSnapshot::read(&path).unwrap().pending_delivery);
    }

    #[tokio::test]
    async fn transcript_popup_review_mode_ignores_expiry() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("transcript.json");
        let publisher = TranscriptPublisher::new(
            path.clone(),
            TranscriptPopupConfig {
                enabled: true,
                review_mode: true,
                final_display_ms: 1,
                ..Default::default()
            },
        )
        .unwrap();
        publisher.stage_for_review("Waiting for approval").unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        let snapshot = TranscriptSnapshot::read(&path).unwrap();
        assert!(snapshot.pending_delivery && snapshot.visible_at(u64::MAX));
    }

    #[test]
    fn transcript_popup_rejects_invalid_appearance() {
        let mut settings = TranscriptPopupConfig {
            opacity: f32::NAN,
            ..Default::default()
        };
        assert!(settings.validate().is_err());
        settings.opacity = 0.8;
        settings.font_size = 0.0;
        assert!(settings.validate().is_err());
    }
}
