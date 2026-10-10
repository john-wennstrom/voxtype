use crate::{alphabet, format::*};
use anyhow::{bail, Context, Result};
use bytemuck::Zeroable;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::Path;

pub struct Entry {
    pub canonical: String,
    pub phones: Vec<u8>,
    pub source: u8,
    pub weight: f32,
}

pub fn from_tsv(variants: &Path, dictionary: &Path, output: &Path) -> Result<()> {
    let mut entries = Vec::new();
    for (line_number, line) in BufReader::new(File::open(variants)?).lines().enumerate() {
        let line = line?;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let columns: Vec<_> = line.split('\t').collect();
        if columns.len() != 5 {
            bail!(
                "{}:{}: expected five TSV columns",
                variants.display(),
                line_number + 1
            );
        }
        let source = match columns[2] {
            "user" => 0,
            "cmudict" => 1,
            "split" => 2,
            "acronym" => 3,
            "g2p_en" => 4,
            "espeak" => 5,
            "observed" => 6,
            source => bail!("Unknown pronunciation source: {source}"),
        };
        entries.push(Entry {
            canonical: columns[1].to_string(),
            phones: alphabet::parse(columns[3])?,
            source,
            weight: columns[4].parse().context("Invalid term weight")?,
        });
    }
    let mut words = BTreeMap::new();
    for line in BufReader::new(File::open(dictionary)?).lines() {
        let line = line?;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (word, phones) = line
            .split_once('\t')
            .context("Expected word TAB phonemes")?;
        words.insert(word.to_lowercase(), alphabet::parse(phones)?);
    }
    write(entries, words, output, 0)
}

pub fn write(
    mut entries: Vec<Entry>,
    dictionary: BTreeMap<String, Vec<u8>>,
    output: &Path,
    tokenizer_hash: u64,
) -> Result<()> {
    if !cfg!(target_endian = "little") {
        bail!("Lexicon indexes require a little-endian host");
    }
    entries.sort_by_key(|entry| entry.phones.len());
    let mut arena = Vec::new();
    let mut variants = Vec::new();
    let mut terms = Vec::new();
    let mut strings = Vec::new();
    let mut term_ids = HashMap::new();
    let mut gram_lists: Vec<Vec<u32>> = vec![Vec::new(); 4096];
    let mut keys: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
    for entry in entries {
        if entry.phones.is_empty()
            || entry.phones.len() > 63
            || entry.phones.iter().any(|phone| *phone >= 39)
            || entry.canonical.is_empty()
            || !entry.weight.is_finite()
            || !(0.0..=1.0).contains(&entry.weight)
            || entry.weight == 0.0
            || entry.source > 6
        {
            bail!("Invalid lexicon entry: {}", entry.canonical);
        }
        let folded = entry.canonical.to_lowercase();
        let term_id = if let Some(&term_id) = term_ids.get(&folded) {
            let term: &Term = &terms[term_id as usize];
            if term.weight != ((entry.weight * u16::MAX as f32).round() as u16).max(1)
                || &strings[term.offset as usize..term.offset as usize + term.length as usize]
                    != entry.canonical.as_bytes()
            {
                bail!("Conflicting spelling or weight for {}", entry.canonical);
            }
            term_id
        } else {
            let term_id = u32::try_from(terms.len())?;
            terms.push(Term {
                offset: u32::try_from(strings.len())?,
                length: u32::try_from(entry.canonical.len())?,
                weight: ((entry.weight * u16::MAX as f32).round() as u16).max(1),
                padding: [0; 6],
            });
            strings.extend_from_slice(entry.canonical.as_bytes());
            term_ids.insert(folded, term_id);
            term_id
        };
        let variant_id = u32::try_from(variants.len())?;
        variants.push(Variant {
            offset: u32::try_from(arena.len())?,
            term: term_id,
            length: entry.phones.len() as u8,
            source: entry.source,
            padding: [0; 2],
        });
        arena.extend_from_slice(&entry.phones);
        let mut grams = [0; 65];
        let count = alphabet::grams(&entry.phones, &mut grams);
        for &gram in &grams[..count] {
            gram_lists[gram as usize].push(variant_id);
        }
        keys.entry(alphabet::key(&entry.phones))
            .or_default()
            .push(variant_id);
    }
    let lengths: Vec<u32> = (0..=64)
        .map(|length| variants.partition_point(|variant| (variant.length as usize) < length) as u32)
        .collect();
    let mut gram_offsets = Vec::with_capacity(4097);
    let mut gram_postings = Vec::new();
    for list in gram_lists {
        gram_offsets.push(u32::try_from(gram_postings.len())?);
        gram_postings.extend(list);
    }
    gram_offsets.push(u32::try_from(gram_postings.len())?);
    let table_size = ((keys.len() * 10 / 7) + 1).next_power_of_two();
    let mut key_table = vec![KeySlot::zeroed(); table_size];
    let mut key_postings = Vec::new();
    for (key, postings) in keys {
        let mut slot = key as usize & (table_size - 1);
        while key_table[slot].length != 0 {
            slot = (slot + 1) & (table_size - 1);
        }
        key_table[slot] = KeySlot {
            key,
            offset: u32::try_from(key_postings.len())?,
            length: u32::try_from(postings.len())?,
        };
        key_postings.extend(postings);
    }
    let mut dict_arena = Vec::new();
    let mut dict_builder = fst::MapBuilder::memory();
    for (word, phones) in dictionary {
        if word.is_empty()
            || phones.is_empty()
            || phones.len() > 63
            || phones.iter().any(|phone| *phone >= 39)
        {
            bail!("Invalid query dictionary entry: {word}");
        }
        let offset = u32::try_from(dict_arena.len())?;
        dict_builder.insert(word, ((offset as u64) << 8) | phones.len() as u64)?;
        dict_arena.extend(phones);
    }
    let dict_bytes = dict_builder.into_inner()?;
    let sections: [&[u8]; SECTIONS] = [
        &arena,
        bytemuck::cast_slice(&variants),
        bytemuck::cast_slice(&lengths),
        bytemuck::cast_slice(&gram_offsets),
        bytemuck::cast_slice(&gram_postings),
        bytemuck::cast_slice(&key_table),
        bytemuck::cast_slice(&key_postings),
        bytemuck::cast_slice(&terms),
        &strings,
        &dict_bytes,
        &dict_arena,
    ];
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    let mut header = Header {
        magic: *b"VOXLEX01",
        version: 1,
        variants: variants.len() as u32,
        terms: terms.len() as u32,
        reserved: 0,
        alphabet: alphabet::hash(),
        tokenizer: tokenizer_hash,
        sections: [Section::zeroed(); SECTIONS],
    };
    file.write_all(bytemuck::bytes_of(&header))?;
    for (position, section) in sections.iter().enumerate() {
        let offset = file.stream_position()?;
        let padding = (64 - offset % 64) % 64;
        file.write_all(&[0; 64][..padding as usize])?;
        header.sections[position] = Section {
            offset: offset + padding,
            length: section.len() as u64,
            checksum: xxhash_rust::xxh3::xxh3_64(section),
        };
        file.write_all(section)?;
    }
    file.seek(SeekFrom::Start(0))?;
    file.write_all(bytemuck::bytes_of(&header))?;
    file.as_file().sync_all()?;
    file.persist(output).map_err(|error| error.error)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}
