use std::path::{Path, PathBuf};

pub(super) fn detect() -> Option<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        let pid = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .ok()?;
                    runtime.block_on(focused_pid())
                })
                .join()
                .ok()
                .flatten()
        })?;
        project_for_pid(Path::new("/proc"), pid)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

async fn focused_pid() -> Option<u32> {
    let (program, arguments): (&str, &[&str]) =
        if std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some() {
            ("hyprctl", &["activewindow", "-j"])
        } else if std::env::var_os("SWAYSOCK").is_some() {
            ("swaymsg", &["-t", "get_tree", "-r"])
        } else if std::env::var("XDG_SESSION_TYPE").is_ok_and(|session| session == "x11") {
            ("xdotool", &["getwindowfocus", "getwindowpid"])
        } else {
            return None;
        };
    let output = tokio::time::timeout(
        std::time::Duration::from_millis(100),
        tokio::process::Command::new(program)
            .args(arguments)
            .kill_on_drop(true)
            .output(),
    )
    .await
    .ok()?
    .ok()?;
    if !output.status.success() || output.stdout.len() > 4_194_304 {
        return None;
    }
    if program == "xdotool" {
        return std::str::from_utf8(&output.stdout)
            .ok()?
            .trim()
            .parse::<u32>()
            .ok()
            .filter(|pid| *pid > 0);
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    if program == "hyprctl" {
        json_pid(&value)
    } else {
        sway_pid(&value)
    }
}

fn json_pid(value: &serde_json::Value) -> Option<u32> {
    value
        .get("pid")?
        .as_u64()
        .and_then(|pid| u32::try_from(pid).ok())
        .filter(|pid| *pid > 0)
}

fn sway_pid(value: &serde_json::Value) -> Option<u32> {
    if value.get("focused").and_then(serde_json::Value::as_bool) == Some(true) {
        if let Some(pid) = json_pid(value) {
            return Some(pid);
        }
    }
    ["nodes", "floating_nodes"]
        .into_iter()
        .filter_map(|key| value.get(key)?.as_array())
        .flatten()
        .find_map(sway_pid)
}

fn project_for_pid(proc_root: &Path, pid: u32) -> Option<PathBuf> {
    let mut queue = vec![(pid, 0)];
    let mut focused_project = None;
    let mut shell_projects = std::collections::HashSet::new();
    let mut saw_shell = false;
    let mut position = 0;
    while position < queue.len() && position < 128 {
        let (current, depth) = queue[position];
        position += 1;
        let process = proc_root.join(current.to_string());
        let shell = std::fs::read_to_string(process.join("comm")).is_ok_and(|name| {
            matches!(
                name.trim(),
                "bash" | "zsh" | "fish" | "sh" | "dash" | "nu" | "ksh"
            )
        });
        if current == pid || shell {
            saw_shell |= shell;
            if let Ok(cwd) = std::fs::read_link(process.join("cwd")) {
                if let Some(found) = voxtype_lexicon::project::discover(&cwd) {
                    if shell {
                        shell_projects.insert(found);
                    } else {
                        focused_project = Some(found);
                    }
                }
            }
        }
        if depth < 8 {
            if let Ok(children) =
                std::fs::read_to_string(process.join(format!("task/{current}/children")))
            {
                for child in children
                    .split_whitespace()
                    .filter_map(|child| child.parse::<u32>().ok())
                {
                    if queue.len() >= 128 {
                        break;
                    }
                    if !queue.iter().any(|(existing, _)| *existing == child) {
                        queue.push((child, depth + 1));
                    }
                }
            }
        }
    }
    if saw_shell {
        if shell_projects.len() == 1 {
            shell_projects.into_iter().next()
        } else {
            None
        }
    } else {
        focused_project
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn focused_floating_sway_window_and_invalid_pids() {
        let tree = serde_json::json!({"nodes": [{"pid": 12, "focused": false}], "floating_nodes": [{"pid": 34, "focused": true}]});
        assert_eq!(sway_pid(&tree), Some(34));
        assert_eq!(json_pid(&serde_json::json!({"pid": -1})), None);
        assert_eq!(json_pid(&serde_json::json!({"pid": 0})), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn terminal_uses_shell_child_directory() {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        std::fs::create_dir_all(project.join(".voxtype")).unwrap();
        std::fs::write(project.join(".voxtype/terms.tsv"), "git").unwrap();
        let proc_root = root.path().join("proc");
        for pid in [10, 11] {
            std::fs::create_dir_all(proc_root.join(format!("{pid}/task/{pid}"))).unwrap();
        }
        std::fs::write(proc_root.join("10/task/10/children"), "11").unwrap();
        std::fs::write(proc_root.join("11/comm"), "bash\n").unwrap();
        std::os::unix::fs::symlink(&project, proc_root.join("11/cwd")).unwrap();
        assert_eq!(project_for_pid(&proc_root, 10), Some(project));
    }
}
