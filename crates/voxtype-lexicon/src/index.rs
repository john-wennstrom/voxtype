use crate::{alphabet, format::*};
use anyhow::{ensure, Context, Result};
use memmap2::Mmap;
use smallvec::SmallVec;
use std::fs::File;
use std::path::Path;

pub struct PhoneticIndex {
    map: Mmap,
    header: Header,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub term: u32,
    pub variant: u32,
    pub cost: u16,
    pub sim: f32,
    pub score: f32,
}

pub struct TermRef<'index> {
    pub canonical: &'index str,
    pub weight: f32,
}

pub struct SearchOpts<'filter> {
    pub max_edits: usize,
    pub limit: usize,
    pub stop_fraction: f32,
    pub exact_fast_path: bool,
    pub excluded_terms: &'filter [bool],
    pub term_weights: &'filter [f32],
}

impl Default for SearchOpts<'_> {
    fn default() -> Self {
        Self {
            max_edits: 3,
            limit: 8,
            stop_fraction: 0.05,
            exact_fast_path: true,
            excluded_terms: &[],
            term_weights: &[],
        }
    }
}

#[derive(Default)]
pub struct Scratch {
    counts: Vec<u16>,
    stamps: Vec<u32>,
    touched: Vec<u32>,
    generation: u32,
}

impl Scratch {
    pub fn new(index: &PhoneticIndex) -> Self {
        let length = index.variant_count();
        Self {
            counts: vec![0; length],
            stamps: vec![0; length],
            touched: Vec::with_capacity(length),
            generation: 0,
        }
    }

    fn begin(&mut self, length: usize) {
        self.counts.resize(length, 0);
        self.stamps.resize(length, 0);
        self.touched.clear();
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.stamps.fill(0);
            self.generation = 1;
        }
    }
}

impl PhoneticIndex {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_tokenizer(path, 0)
    }

    pub fn open_with_tokenizer(path: &Path, tokenizer_hash: u64) -> Result<Self> {
        ensure!(
            cfg!(target_endian = "little"),
            "Lexicon indexes require little endian"
        );
        let file = File::open(path).with_context(|| format!("Open lexicon {}", path.display()))?;
        let map = unsafe { Mmap::map(&file)? };
        ensure!(
            map.len() >= std::mem::size_of::<Header>(),
            "Truncated lexicon header"
        );
        let header = *bytemuck::try_from_bytes::<Header>(&map[..std::mem::size_of::<Header>()])
            .map_err(|error| anyhow::anyhow!("Invalid header alignment: {error}"))?;
        ensure!(
            header.magic == *b"VOXLEX01" && header.version == 1,
            "Unsupported lexicon format"
        );
        ensure!(
            header.alphabet == alphabet::hash(),
            "Stale phoneme alphabet; rebuild lexicon"
        );
        ensure!(
            header.tokenizer == tokenizer_hash,
            "Stale tokenizer hash; rebuild lexicon"
        );
        let mut end = std::mem::size_of::<Header>();
        for section in &header.sections {
            let offset = usize::try_from(section.offset)?;
            let length = usize::try_from(section.length)?;
            ensure!(
                offset % 64 == 0 && offset >= end,
                "Invalid section alignment or overlap"
            );
            end = offset.checked_add(length).context("Section overflow")?;
            ensure!(end <= map.len(), "Truncated lexicon section");
            ensure!(
                xxhash_rust::xxh3::xxh3_64(&map[offset..end]) == section.checksum,
                "Lexicon checksum mismatch"
            );
        }
        let index = Self { map, header };
        index.validate()?;
        Ok(index)
    }

    fn bytes(&self, section: usize) -> &[u8] {
        let section = self.header.sections[section];
        &self.map[section.offset as usize..(section.offset + section.length) as usize]
    }

    fn slice<T: bytemuck::Pod>(&self, section: usize) -> &[T] {
        bytemuck::cast_slice(self.bytes(section))
    }

    fn validate(&self) -> Result<()> {
        for (section, size) in [
            (VARIANTS, 12),
            (LENGTHS, 4),
            (GRAM_OFFSETS, 4),
            (GRAM_POSTINGS, 4),
            (KEYS, 16),
            (KEY_POSTINGS, 4),
            (TERMS, 16),
        ] {
            ensure!(
                self.bytes(section).len().is_multiple_of(size),
                "Invalid lexicon record size"
            );
        }
        let variants = self.slice::<Variant>(VARIANTS);
        let terms = self.slice::<Term>(TERMS);
        ensure!(
            variants.len() == self.header.variants as usize
                && terms.len() == self.header.terms as usize,
            "Count mismatch"
        );
        ensure!(
            self.bytes(ARENA)
                .iter()
                .chain(self.bytes(DICT_ARENA))
                .all(|phone| *phone < 39),
            "Invalid phoneme id"
        );
        let strings = std::str::from_utf8(self.bytes(STRINGS))?;
        for term in terms {
            ensure!(
                strings
                    .get(term.offset as usize..term.offset as usize + term.length as usize)
                    .is_some_and(|text| !text.is_empty()),
                "Invalid canonical string"
            );
            ensure!(term.weight > 0, "Invalid term weight");
        }
        for variant in variants {
            ensure!(
                (1..=63).contains(&variant.length)
                    && variant.source <= 6
                    && (variant.term as usize) < terms.len(),
                "Invalid variant"
            );
            ensure!(
                variant.offset as usize + variant.length as usize <= self.bytes(ARENA).len(),
                "Invalid phoneme range"
            );
        }
        ensure!(
            variants
                .windows(2)
                .all(|pair| pair[0].length <= pair[1].length),
            "Unsorted variants"
        );
        let lengths = self.slice::<u32>(LENGTHS);
        ensure!(lengths.len() == 65, "Invalid length table");
        for (length, &start) in lengths.iter().enumerate() {
            ensure!(
                start as usize
                    == variants.partition_point(|variant| (variant.length as usize) < length),
                "Invalid length range"
            );
        }
        let offsets = self.slice::<u32>(GRAM_OFFSETS);
        let postings = self.slice::<u32>(GRAM_POSTINGS);
        ensure!(
            offsets.len() == 4097 && offsets[0] == 0 && offsets[4096] as usize == postings.len(),
            "Invalid q-gram offsets"
        );
        ensure!(
            offsets.windows(2).all(|pair| pair[0] <= pair[1]),
            "Unsorted q-gram offsets"
        );
        ensure!(
            postings
                .iter()
                .chain(self.slice::<u32>(KEY_POSTINGS))
                .all(|id| (*id as usize) < variants.len()),
            "Invalid posting id"
        );
        for range in offsets.windows(2) {
            ensure!(
                postings[range[0] as usize..range[1] as usize]
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
                "Unsorted postings"
            );
        }
        let keys = self.slice::<KeySlot>(KEYS);
        ensure!(
            keys.len().is_power_of_two() && keys.iter().any(|slot| slot.length == 0),
            "Invalid key table"
        );
        for slot in keys {
            ensure!(
                slot.offset as usize + slot.length as usize
                    <= self.slice::<u32>(KEY_POSTINGS).len(),
                "Invalid key posting range"
            );
        }
        let dictionary = fst::Map::new(self.bytes(DICTIONARY))?;
        let mut stream = dictionary.stream();
        use fst::Streamer;
        while let Some((word, value)) = stream.next() {
            std::str::from_utf8(word)?;
            let length = (value & 255) as usize;
            ensure!(
                (1..=63).contains(&length)
                    && (value >> 8) <= self.bytes(DICT_ARENA).len() as u64
                    && length <= self.bytes(DICT_ARENA).len() - (value >> 8) as usize,
                "Invalid dictionary phoneme range"
            );
        }
        Ok(())
    }

    pub fn variant_count(&self) -> usize {
        self.header.variants as usize
    }

    pub fn term_count(&self) -> usize {
        self.header.terms as usize
    }

    pub fn term(&self, id: u32) -> TermRef<'_> {
        let term = &self.slice::<Term>(TERMS)[id as usize];
        let canonical = std::str::from_utf8(
            &self.bytes(STRINGS)[term.offset as usize..term.offset as usize + term.length as usize],
        )
        .expect("validated string");
        TermRef {
            canonical,
            weight: term.weight as f32 / u16::MAX as f32,
        }
    }

    pub fn g2p_word(&self, word: &str, output: &mut SmallVec<[u8; 32]>) -> bool {
        let mut normalized = [0u8; 128];
        let word =
            word.trim_matches(|character: char| !character.is_alphanumeric() && character != '\'');
        if word.is_empty() || word.len() > normalized.len() || !word.is_ascii() {
            return false;
        }
        for (target, byte) in normalized.iter_mut().zip(word.bytes()) {
            *target = byte.to_ascii_lowercase();
        }
        let dictionary = fst::Map::new(self.bytes(DICTIONARY)).expect("validated dictionary");
        let Some(value) = dictionary.get(&normalized[..word.len()]) else {
            return false;
        };
        let offset = (value >> 8) as usize;
        let length = (value & 255) as usize;
        output.extend_from_slice(&self.bytes(DICT_ARENA)[offset..offset + length]);
        true
    }

    fn phones(&self, variant: &Variant) -> &[u8] {
        &self.bytes(ARENA)
            [variant.offset as usize..variant.offset as usize + variant.length as usize]
    }

    fn verify(&self, id: u32, phones: &[u8], options: &SearchOpts, output: &mut Vec<Candidate>) {
        let variant = &self.slice::<Variant>(VARIANTS)[id as usize];
        if options
            .excluded_terms
            .get(variant.term as usize)
            .copied()
            .unwrap_or(false)
        {
            return;
        }
        let max_cost = options.max_edits.min(63) as u16 * 4;
        let cost = alphabet::distance(phones, self.phones(variant), max_cost);
        if cost > max_cost {
            return;
        }
        let sim =
            (1.0 - cost as f32 / (phones.len().max(variant.length as usize) * 4) as f32).max(0.0);
        let trust = [1.0, 0.98, 0.96, 0.96, 0.90, 0.90, 1.0][variant.source as usize];
        let candidate = Candidate {
            term: variant.term,
            variant: id,
            cost,
            sim,
            score: sim
                * trust
                * options
                    .term_weights
                    .get(variant.term as usize)
                    .copied()
                    .unwrap_or_else(|| self.term(variant.term).weight),
        };
        if let Some(position) = output
            .iter()
            .position(|existing| existing.term == candidate.term)
        {
            if output[position].score >= candidate.score {
                return;
            }
            output.remove(position);
        }
        let position = output.partition_point(|existing| {
            existing.score > candidate.score
                || (existing.score == candidate.score && existing.term < candidate.term)
        });
        if position < options.limit {
            if output.len() == options.limit {
                output.pop();
            }
            output.insert(position, candidate);
        }
    }

    pub fn search(
        &self,
        phones: &[u8],
        options: &SearchOpts,
        scratch: &mut Scratch,
        output: &mut Vec<Candidate>,
    ) {
        output.clear();
        if phones.is_empty()
            || phones.len() > 63
            || phones.iter().any(|phone| *phone >= 39)
            || options.limit == 0
        {
            return;
        }
        scratch.begin(self.variant_count());
        let key = alphabet::key(phones);
        let slots = self.slice::<KeySlot>(KEYS);
        let mut position = key as usize & (slots.len() - 1);
        for _ in 0..slots.len() {
            let slot = &slots[position];
            if slot.length == 0 {
                break;
            }
            if slot.key == key {
                for &id in &self.slice::<u32>(KEY_POSTINGS)
                    [slot.offset as usize..slot.offset as usize + slot.length as usize]
                {
                    self.verify(id, phones, options, output);
                }
                break;
            }
            position = (position + 1) & (slots.len() - 1);
        }
        if options.exact_fast_path
            && output
                .iter()
                .any(|candidate| candidate.cost == 0 && candidate.score >= 0.95)
        {
            return;
        }
        let mut grams = [0; 65];
        let gram_count = alphabet::grams(phones, &mut grams);
        let lengths = self.slice::<u32>(LENGTHS);
        let low = lengths[phones.len().saturating_sub(options.max_edits)];
        let high = lengths[phones
            .len()
            .saturating_add(options.max_edits)
            .saturating_add(1)
            .min(64)];
        let offsets = self.slice::<u32>(GRAM_OFFSETS);
        let postings = self.slice::<u32>(GRAM_POSTINGS);
        let mut used = 0usize;
        for &gram in &grams[..gram_count] {
            let list =
                &postings[offsets[gram as usize] as usize..offsets[gram as usize + 1] as usize];
            if list.len() as f32 > self.variant_count() as f32 * options.stop_fraction {
                continue;
            }
            used += 1;
            let start = list.partition_point(|id| *id < low);
            let end = list.partition_point(|id| *id < high);
            for &id in &list[start..end] {
                let position = id as usize;
                if scratch.stamps[position] != scratch.generation {
                    scratch.stamps[position] = scratch.generation;
                    scratch.counts[position] = 0;
                    scratch.touched.push(id);
                }
                scratch.counts[position] += 1;
            }
        }
        let threshold = used.saturating_sub(options.max_edits.saturating_mul(3));
        if threshold == 0 {
            for id in low..high {
                self.verify(id, phones, options, output);
            }
        } else {
            for &id in &scratch.touched {
                if scratch.counts[id as usize] as usize >= threshold {
                    self.verify(id, phones, options, output);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{self, Entry};
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    fn create(phones: Vec<u8>, tokenizer: u64) -> (tempfile::TempDir, PhoneticIndex) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("test.vxl");
        build::write(
            vec![Entry {
                canonical: "kubectl".into(),
                phones: phones.clone(),
                source: 0,
                weight: 1.0,
            }],
            BTreeMap::from([("cube".into(), phones)]),
            &path,
            tokenizer,
        )
        .unwrap();
        let index = PhoneticIndex::open_with_tokenizer(&path, tokenizer).unwrap();
        (directory, index)
    }

    #[test]
    fn round_trip_and_corruption() {
        let phones = alphabet::parse("K Y UW B K AH N T R OW L").unwrap();
        let (directory, index) = create(phones.clone(), 42);
        let mut output = Vec::with_capacity(8);
        index.search(
            &phones,
            &SearchOpts::default(),
            &mut Scratch::new(&index),
            &mut output,
        );
        assert_eq!(output[0].sim, 1.0);
        assert_eq!(index.term(output[0].term).canonical, "kubectl");
        let mut phonemes = SmallVec::new();
        assert!(index.g2p_word("Cube,", &mut phonemes));
        assert_eq!(&phonemes[..], &phones);
        let path = directory.path().join("test.vxl");
        assert!(PhoneticIndex::open(&path).is_err());
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] ^= 1;
        let corrupt = directory.path().join("corrupt.vxl");
        std::fs::write(&corrupt, bytes).unwrap();
        assert!(PhoneticIndex::open_with_tokenizer(&corrupt, 42).is_err());
        for length in [0, 16, 303] {
            std::fs::write(&corrupt, vec![0; length]).unwrap();
            assert!(PhoneticIndex::open(&corrupt).is_err());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn fuzzy_search_matches_brute_force(phones in prop::collection::vec(0u8..39, 4..30), replacement in 0u8..39, position in 0usize..30) {
            let (_directory, index) = create(phones.clone(), 0);
            let mut query = phones.clone();
            let position = position % query.len();
            query[position] = replacement;
            let mut output = Vec::with_capacity(8);
            index.search(&query, &SearchOpts { max_edits: 1, stop_fraction: 1.0, ..Default::default() }, &mut Scratch::new(&index), &mut output);
            prop_assert_eq!(output.len(), 1);
            prop_assert_eq!(output[0].cost, alphabet::distance(&query, &phones, 4));
        }
        #[test]
        fn retrieval_preserves_insertions_and_deletions(phones in prop::collection::vec(0u8..39, 4..30), inserted in 0u8..39, position in 0usize..30) {
            let (_directory, index) = create(phones.clone(), 0);
            let position = position % phones.len();
            let mut insertion = phones.clone();
            insertion.insert(position, inserted);
            let mut deletion = phones.clone();
            deletion.remove(position);
            for query in [insertion, deletion] {
                let mut output = Vec::with_capacity(8);
                index.search(&query, &SearchOpts { max_edits: 1, stop_fraction: 1.0, exact_fast_path: false, ..Default::default() }, &mut Scratch::new(&index), &mut output);
                prop_assert_eq!(output.len(), 1);
                prop_assert_eq!(output[0].cost, 3);
            }
        }
        #[test]
        fn arbitrary_bytes_are_rejected_without_panicking(bytes in prop::collection::vec(any::<u8>(), 0..4096)) {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("corrupt.vxl");
            std::fs::write(&path, bytes).unwrap();
            prop_assert!(PhoneticIndex::open(&path).is_err());
        }
    }
}
