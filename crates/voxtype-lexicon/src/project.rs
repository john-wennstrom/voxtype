use std::path::{Path, PathBuf};

pub fn discover(directory: &Path) -> Option<PathBuf> {
    directory
        .ancestors()
        .find(|directory| directory.join(".voxtype/terms.tsv").is_file())
        .map(Path::to_path_buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_project_wins_and_missing_projects_are_ignored() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("nested");
        std::fs::create_dir_all(nested.join("src")).unwrap();
        assert!(discover(&nested).is_none());
        for directory in [root.path(), nested.as_path()] {
            std::fs::create_dir_all(directory.join(".voxtype")).unwrap();
            std::fs::write(directory.join(".voxtype/terms.tsv"), "git\n").unwrap();
        }
        assert_eq!(discover(&nested.join("src")), Some(nested));
    }
}
