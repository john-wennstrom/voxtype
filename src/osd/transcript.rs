use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct TranscriptPopupConfig {
    pub enabled: bool,
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

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TranscriptSnapshot {
    pub text: String,
    pub recording: bool,
    pub expires_at_ms: u64,
    pub settings: TranscriptPopupConfig,
}

impl TranscriptSnapshot {
    pub fn visible_at(&self, now_ms: u64) -> bool {
        self.settings.enabled && (self.recording || now_ms < self.expires_at_ms)
    }

    pub fn read(path: &Path) -> Option<Self> {
        serde_json::from_slice(&std::fs::read(path).ok()?).ok()
    }
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
        if !self.settings.enabled {
            return Ok(());
        }
        let mut revision = self.revision.lock().expect("popup revision poisoned");
        *revision += 1;
        let published_revision = *revision;
        let snapshot = TranscriptSnapshot {
            text: text.to_string(),
            recording,
            expires_at_ms: now_ms().saturating_add(self.settings.final_display_ms),
            settings: self.settings.clone(),
        };
        let directory = self.path.parent().expect("validated popup directory");
        let mut temporary = tempfile::NamedTempFile::new_in(directory)?;
        serde_json::to_writer(&mut temporary, &snapshot)?;
        temporary.flush()?;
        temporary.persist(&self.path).map_err(|error| error.error)?;
        if !recording {
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
        Ok(())
    }

    pub fn clear(&self) {
        let mut revision = self.revision.lock().expect("popup revision poisoned");
        *revision += 1;
        let _ = std::fs::remove_file(&self.path);
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
        publisher.publish("Live transcript preview. Text wraps here without stealing focus or typing into your editor.", true).unwrap();
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
