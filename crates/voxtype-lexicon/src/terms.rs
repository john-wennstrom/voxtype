use anyhow::{bail, ensure, Result};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

pub fn normalize(text: &str) -> String {
    text.split_whitespace()
        .map(|word| {
            word.trim_matches(|character: char| !character.is_alphanumeric())
                .to_lowercase()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn aliases(source: &str) -> Result<HashMap<String, String>> {
    let mut aliases = HashMap::new();
    for line in source.lines() {
        if line.trim().is_empty() || line.trim_start().starts_with(['#', '!']) {
            continue;
        }
        let columns: Vec<_> = line.split('\t').collect();
        let canonical = columns[0].trim();
        for form in columns
            .get(1)
            .into_iter()
            .flat_map(|forms| forms.split('|'))
        {
            let key = normalize(form);
            if key.is_empty() || key == normalize(canonical) {
                continue;
            }
            if let Some(previous) = aliases.insert(key.clone(), canonical.to_string()) {
                ensure!(
                    previous == canonical,
                    "Conflicting spoken form {key}: {previous} and {canonical}"
                );
            }
        }
    }
    Ok(aliases)
}

pub fn corrected_source(source: &str, recognized: &str, canonical: &str) -> Result<String> {
    for text in [recognized, canonical] {
        ensure!(
            !text.trim().is_empty() && !text.contains(['\t', '\r', '\n', '|']),
            "Correction must be nonempty and cannot contain TSV separators"
        );
    }
    ensure!(
        !canonical.trim_start().starts_with(['#', '!']),
        "Canonical term cannot start with # or !"
    );
    ensure!(
        (1..=3).contains(&recognized.split_whitespace().count()),
        "Select one to three words to learn a term"
    );
    let key = normalize(recognized);
    ensure!(!key.is_empty(), "Correction must include a word");
    if let Some(previous) = aliases(source)?.get(&key) {
        ensure!(
            previous.eq_ignore_ascii_case(canonical.trim()),
            "Spoken form already belongs to {previous}"
        );
    }
    let mut lines: Vec<_> = source.lines().map(str::to_owned).collect();
    let mut found = false;
    for line in &mut lines {
        if line.trim_start().starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut columns: Vec<_> = line.split('\t').map(str::to_owned).collect();
        if columns[0]
            .trim()
            .strip_prefix('!')
            .is_some_and(|term| term.eq_ignore_ascii_case(canonical.trim()))
        {
            bail!("Term is suppressed; remove its ! entry before learning it");
        }
        if !columns[0].trim().eq_ignore_ascii_case(canonical.trim()) {
            continue;
        }
        found = true;
        columns.resize(columns.len().max(2), String::new());
        if !columns[1].split('|').any(|form| normalize(form) == key) {
            if !columns[1].is_empty() {
                columns[1].push('|');
            }
            columns[1].push_str(recognized.trim());
        }
        *line = columns.join("\t");
        break;
    }
    if !found {
        lines.push(format!("{}\t{}", canonical.trim(), recognized.trim()));
    }
    let output = format!("{}\n", lines.join("\n"));
    aliases(&output)?;
    Ok(output)
}

pub fn save_correction(path: &Path, recognized: &str, canonical: &str) -> Result<()> {
    let source = std::fs::read_to_string(path)?;
    let corrected = corrected_source(&source, recognized, canonical)?;
    if corrected == source {
        return Ok(());
    }
    let mut temporary = tempfile::NamedTempFile::new_in(path.parent().unwrap_or(Path::new(".")))?;
    temporary.write_all(corrected.as_bytes())?;
    temporary.as_file().sync_all()?;
    ensure!(
        std::fs::read_to_string(path)? == source,
        "Terms changed while saving; retry the correction"
    );
    temporary.persist(path).map_err(|error| error.error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reviewed_terms_preserve_forms_weights_and_reject_conflicts() {
        let source = "# private terms\nVoxtype\tvox type\t0.9\n!suppressed\n";
        let updated = corrected_source(source, "Vuxtype", "Voxtype").unwrap();
        assert_eq!(
            updated,
            "# private terms\nVoxtype\tvox type|Vuxtype\t0.9\n!suppressed\n"
        );
        assert_eq!(
            corrected_source(&updated, "Vuxtype", "Voxtype").unwrap(),
            updated
        );
        assert_eq!(aliases(&updated).unwrap()["vuxtype"], "Voxtype");
        assert!(corrected_source(&updated, "Vuxtype", "Other").is_err());
        assert!(corrected_source(source, "word", "suppressed").is_err());
        assert!(corrected_source(source, "bad\trow", "Name").is_err());
        assert!(corrected_source(source, "one two three four", "Name").is_err());
        assert!(corrected_source(source, "word", "!Name").is_err());
        assert!(corrected_source(source, "new phrase", "NewName")
            .unwrap()
            .ends_with("NewName\tnew phrase\n"));
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("terms.tsv");
        std::fs::write(&path, source).unwrap();
        save_correction(&path, "Vuxtype", "Voxtype").unwrap();
        assert_eq!(std::fs::read_to_string(path).unwrap(), updated);
    }
}
