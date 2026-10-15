use anyhow::{ensure, Context, Result};
use std::io::{BufRead, Write};
use std::path::Path;
use voxtype_lexicon::evaluation::{word_errors, Sample};

const STARTER_WORDS: &str = "# Project vocabulary: one word or phrase per line.\n# Optional: canonical term<TAB>spoken form|alternate form<TAB>weight (0..1).\n# Replace these examples, then rerun voxtype init.\nVoxtype\tvoice type\nProjectName\tproject name\n";

fn prepare_project(project: &Path) -> Result<std::path::PathBuf> {
    let project = project
        .canonicalize()
        .context("Project directory must exist; run voxtype init from your project directory")?;
    ensure!(project.is_dir(), "Project path must be a directory");
    let directory = project.join(".voxtype");
    std::fs::create_dir_all(&directory)?;
    let words = directory.join("words.txt");
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&words)
    {
        Ok(mut file) => {
            let legacy = directory.join("terms.tsv");
            if legacy.exists() {
                file.write_all(&std::fs::read(legacy)?)?;
            } else {
                file.write_all(STARTER_WORDS.as_bytes())?;
            }
            file.sync_all()?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
        Err(error) => return Err(error.into()),
    }
    Ok(project)
}

fn run_setup(command: &mut std::process::Command, step: &str) -> Result<()> {
    let status = command.status().with_context(|| {
        format!(
            "{step}: could not run {}; install Python 3 with venv and pip support, or use --python",
            command.get_program().to_string_lossy()
        )
    })?;
    ensure!(
        status.success(),
        "{step} failed ({status}); fix the error above and rerun voxtype init"
    );
    Ok(())
}

fn ensure_user_index(query: &Path, index: &Path) -> Result<()> {
    if !index.try_exists()? {
        let empty = tempfile::NamedTempFile::new()?;
        voxtype_lexicon::build::from_tsv(empty.path(), query, index)?;
    }
    Ok(())
}

pub(super) fn init(
    project: &Path,
    backend: &str,
    python: &Path,
    settings: &voxtype::config::LexiconConfig,
) -> Result<()> {
    let project = prepare_project(project)?;
    let directory = project.join(".voxtype");
    let cache = dirs::cache_dir().context("Cannot locate the user cache directory")?;
    let runtime = cache.join("voxtype/lexicon-tools");
    std::fs::create_dir_all(&runtime)?;
    let requirements = runtime.join("requirements.txt");
    let requirements_text = include_str!("../../tools/requirements-lexicon.txt");
    std::fs::write(&requirements, requirements_text)?;
    let script = runtime.join("lexicon_g2p.py");
    std::fs::write(&script, include_str!("../../tools/lexicon_g2p.py"))?;
    let environment = runtime.join("venv");
    let interpreter = environment.join("bin/python");
    if !interpreter.is_file() {
        println!(
            "Creating shared lexicon Python environment: {}",
            environment.display()
        );
        run_setup(
            std::process::Command::new(python)
                .arg("-m")
                .arg("venv")
                .arg(&environment),
            "Creating lexicon environment",
        )?;
    }
    let installed = runtime.join("installed-requirements.txt");
    if std::fs::read_to_string(&installed).ok().as_deref() != Some(requirements_text) {
        run_setup(
            std::process::Command::new(&interpreter)
                .args(["-m", "pip", "install", "-r"])
                .arg(&requirements),
            "Installing lexicon dependencies",
        )?;
        std::fs::write(installed, requirements_text)?;
    }
    let nltk_data = std::env::var_os("VOXTYPE_NLTK_DATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| cache.join("voxtype-nltk-data"));
    let mut directory_builder = std::fs::DirBuilder::new();
    directory_builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        directory_builder.mode(0o700);
    }
    directory_builder.create(&nltk_data)?;
    run_setup(
        std::process::Command::new(&interpreter).env("NLTK_DATA", &nltk_data).args([
            "-c",
            "import os, nltk\nnltk.data.path[:] = [os.environ['NLTK_DATA']]\nfor resource, package in [('taggers/averaged_perceptron_tagger.zip', 'averaged_perceptron_tagger'), ('taggers/averaged_perceptron_tagger_eng/', 'averaged_perceptron_tagger_eng'), ('corpora/cmudict.zip', 'cmudict')]:\n try: nltk.data.find(resource)\n except LookupError:\n  nltk.download(package, download_dir=os.environ['NLTK_DATA'], raise_on_error=True)\n  nltk.data.find(resource)",
        ]),
        "Preparing NLTK data",
    ).context("NLTK data must be in a private directory; set VOXTYPE_NLTK_DATA to a directory whose ancestors are not group- or world-writable")?;
    let terms = directory.join("terms.tsv");
    let mut source = tempfile::NamedTempFile::new_in(&directory)?;
    source.write_all(&std::fs::read(directory.join("words.txt"))?)?;
    source.as_file().sync_all()?;
    source.persist(&terms).map_err(|error| error.error)?;
    let variants = directory.join("terms.phon.tsv");
    let query = directory.join("querydict.phon.tsv");
    run_setup(
        std::process::Command::new(&interpreter)
            .env("NLTK_DATA", nltk_data)
            .arg(script)
            .arg(&terms)
            .arg(&variants)
            .arg("--backend")
            .arg(backend)
            .arg("--export-query-dict")
            .arg(&query),
        "Generating project pronunciations",
    )?;
    let index =
        voxtype::config::LexiconConfig::project_index_path(&project).map_err(anyhow::Error::msg)?;
    voxtype_lexicon::build::from_tsv_with_corrections(&variants, &query, &index, None)?;
    ensure_user_index(&query, &settings.index_path().map_err(anyhow::Error::msg)?)?;
    select_project(&project)?;
    println!(
        "Project lexicon ready. Edit {}, then rerun voxtype init.",
        directory.join("words.txt").display()
    );
    println!("Enable corrections with voxtype config set lexicon.enabled true (restart a running daemon).");
    Ok(())
}

pub(super) fn select_project(project: &Path) -> Result<()> {
    let project = project.canonicalize()?;
    ensure!(
        project.join(".voxtype/terms.tsv").is_file(),
        "Project must contain .voxtype/terms.tsv"
    );
    let index =
        voxtype::config::LexiconConfig::project_index_path(&project).map_err(anyhow::Error::msg)?;
    voxtype_lexicon::PhoneticIndex::open(&index)?;
    let active_file =
        voxtype::config::LexiconConfig::active_project_file().map_err(anyhow::Error::msg)?;
    let parent = active_file
        .parent()
        .context("Invalid active-project path")?;
    std::fs::create_dir_all(parent)?;
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    writeln!(
        file,
        "{}",
        project.to_str().context("Project path must be UTF-8")?
    )?;
    file.as_file().sync_all()?;
    file.persist(active_file).map_err(|error| error.error)?;
    println!("Selected project lexicon: {}", project.display());
    Ok(())
}

pub(super) fn synthesize(terms: &Path, output: &Path, voices: &[String]) -> Result<()> {
    let program = which::which("espeak-ng")
        .or_else(|_| which::which("espeak"))
        .context("Install espeak-ng to generate synthetic lexicon audio")?;
    ensure!(!voices.is_empty(), "Provide at least one TTS voice");
    std::fs::create_dir_all(output)?;
    let output = output.canonicalize()?;
    let mut manifest = tempfile::NamedTempFile::new_in(&output)?;
    let mut count = 0;
    for line in std::fs::read_to_string(terms)?.lines() {
        if line.trim().is_empty() || line.trim_start().starts_with(['#', '!']) {
            continue;
        }
        let columns: Vec<_> = line.split('\t').collect();
        let canonical = columns[0].trim();
        let spoken = columns
            .get(1)
            .filter(|forms| !forms.trim().is_empty())
            .map(|forms| forms.split('|').next().unwrap().trim())
            .unwrap_or(canonical);
        for voice in voices {
            for (prefix, suffix, speed) in [
                ("then run", "on the cluster", 150),
                ("we use", "in this project", 175),
                ("please check", "before deployment", 200),
            ] {
                let audio = output.join(format!("sample-{count:06}.wav"));
                ensure!(
                    !audio.exists(),
                    "Synthetic audio already exists: {}; choose a fresh output directory",
                    audio.display()
                );
                let temporary = tempfile::NamedTempFile::new_in(&output)?;
                let status = std::process::Command::new(&program)
                    .args(["-v", voice, "-s", &speed.to_string(), "-w"])
                    .arg(temporary.path())
                    .arg(format!("{prefix} {spoken} {suffix}"))
                    .status()?;
                ensure!(
                    status.success(),
                    "TTS failed for {canonical} with voice {voice}"
                );
                temporary.persist(&audio).map_err(|error| error.error)?;
                serde_json::to_writer(
                    &mut manifest,
                    &Sample {
                        audio,
                        canonical: canonical.into(),
                        spoken: spoken.into(),
                        prefix: prefix.into(),
                        suffix: suffix.into(),
                    },
                )?;
                writeln!(manifest)?;
                count += 1;
            }
        }
    }
    ensure!(count > 0, "No terms to synthesize");
    manifest.as_file().sync_all()?;
    manifest
        .persist(output.join("manifest.jsonl"))
        .map_err(|error| error.error)?;
    println!(
        "Generated {count} samples: {}",
        output.join("manifest.jsonl").display()
    );
    Ok(())
}

pub(super) fn evaluate(
    config: &voxtype::config::Config,
    manifest: &Path,
    corrections: &Path,
) -> Result<()> {
    let mut config = config.clone();
    config.lexicon.enabled = false;
    let transcriber = voxtype::transcribe::create_transcriber(&config)?;
    let parent = corrections
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let mut output = tempfile::NamedTempFile::new_in(parent)?;
    writeln!(output, "# Review these recognized-text TAB canonical-term pairs before using lexicon build --corrections")?;
    let (mut total, mut hits, mut errors, mut reference_words, mut skipped) = (0, 0, 0, 0, 0);
    for line in std::io::BufReader::new(std::fs::File::open(manifest)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let sample: Sample =
            serde_json::from_str(&line).context("Invalid synthetic manifest sample")?;
        let path = if sample.audio.is_absolute() {
            sample.audio.clone()
        } else {
            manifest
                .parent()
                .unwrap_or(Path::new("."))
                .join(&sample.audio)
        };
        let reader = hound::WavReader::open(&path)?;
        let spec = reader.spec();
        ensure!(
            spec.channels > 0 && spec.sample_rate > 0,
            "Invalid WAV format"
        );
        let samples = match spec.sample_format {
            hound::SampleFormat::Int => reader
                .into_samples::<i32>()
                .map(|sample| {
                    sample.map(|sample| sample as f32 / 2f32.powi(spec.bits_per_sample as i32 - 1))
                })
                .collect::<Result<Vec<_>, _>>()?,
            hound::SampleFormat::Float => reader
                .into_samples::<f32>()
                .collect::<Result<Vec<_>, _>>()?,
        };
        let mono: Vec<_> = samples
            .chunks(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect();
        let audio = voxtype::audio::resampler::resample_buffer(&mono, spec.sample_rate, 16000);
        let transcript = transcriber.transcribe(&audio)?;
        let (sample_errors, sample_words) = word_errors(&sample.reference(), &transcript);
        errors += sample_errors;
        reference_words += sample_words;
        total += 1;
        match sample.recognized_term(&transcript) {
            Some(recognized) if recognized.eq_ignore_ascii_case(&sample.canonical) => hits += 1,
            Some(recognized) => {
                writeln!(output, "{recognized}\t{}", sample.canonical)?;
            }
            None => skipped += 1,
        }
        println!("{}\t{}", path.display(), transcript);
    }
    ensure!(
        total > 0 && reference_words > 0,
        "Manifest has no evaluation samples"
    );
    output.as_file().sync_all()?;
    output.persist(corrections).map_err(|error| error.error)?;
    println!(
        "Samples: {total}; term recall: {:.2}%; WER: {:.2}%; unaligned (not learned): {skipped}",
        hits as f64 * 100.0 / total as f64,
        errors as f64 * 100.0 / reference_words as f64
    );
    println!("Review correction pairs: {}", corrections.display());
    Ok(())
}

#[cfg(test)]
mod init_tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn init_provisions_missing_user_index_and_preserves_existing_vocabulary() {
        let root = tempfile::tempdir().unwrap();
        let query = root.path().join("query.tsv");
        let user_index = root.path().join("cache/user.vxl");
        std::fs::write(&query, "project\tP R AA JH EH K T\n").unwrap();
        ensure_user_index(&query, &user_index).unwrap();
        assert_eq!(
            voxtype_lexicon::PhoneticIndex::open(&user_index)
                .unwrap()
                .term_count(),
            0
        );
        let variants = root.path().join("variants.tsv");
        std::fs::write(&variants, "0\tExistingTerm\tuser\tT ER M\t1\n").unwrap();
        voxtype_lexicon::build::from_tsv(&variants, &query, &user_index).unwrap();
        let before = std::fs::read(&user_index).unwrap();
        ensure_user_index(&query, &user_index).unwrap();
        assert_eq!(std::fs::read(&user_index).unwrap(), before);
    }

    #[test]
    fn init_defaults_to_current_project() {
        let cli = voxtype::Cli::try_parse_from(["voxtype", "init"]).unwrap();
        assert!(
            matches!(cli.command, Some(voxtype::Commands::Init { project, backend, python })
            if project == Path::new(".") && backend == "g2p_en" && python == Path::new("python3"))
        );
    }

    #[test]
    fn init_preserves_edited_words_in_another_project() {
        let project = tempfile::tempdir().unwrap();
        let path = prepare_project(project.path()).unwrap();
        let words = path.join(".voxtype/words.txt");
        assert_eq!(std::fs::read_to_string(&words).unwrap(), STARTER_WORDS);
        std::fs::write(&words, "MyProject\tmy project\n").unwrap();
        prepare_project(project.path()).unwrap();
        assert_eq!(
            std::fs::read_to_string(words).unwrap(),
            "MyProject\tmy project\n"
        );
    }

    #[test]
    fn init_imports_existing_terms_without_overwriting_them() {
        let project = tempfile::tempdir().unwrap();
        std::fs::create_dir(project.path().join(".voxtype")).unwrap();
        let terms = project.path().join(".voxtype/terms.tsv");
        std::fs::write(&terms, "ExistingTerm\n").unwrap();
        let path = prepare_project(project.path()).unwrap();
        assert_eq!(
            std::fs::read_to_string(path.join(".voxtype/words.txt")).unwrap(),
            "ExistingTerm\n"
        );
        assert_eq!(std::fs::read_to_string(terms).unwrap(), "ExistingTerm\n");
    }
}
