use crate::{alphabet, build::Entry};
use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, HashSet};

pub fn learn(
    entries: &mut Vec<Entry>,
    dictionary: &BTreeMap<String, Vec<u8>>,
    pairs: &str,
) -> Result<alphabet::Costs> {
    let mut counts = [[0u32; 40]; 40];
    let mut seen: HashSet<_> = entries
        .iter()
        .map(|entry| (entry.canonical.to_lowercase(), entry.phones.clone()))
        .collect();
    for (position, line) in pairs.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let (recognized, canonical) = line.split_once('\t').with_context(|| {
            format!(
                "Correction {}: expected recognized text TAB canonical term",
                position + 1
            )
        })?;
        let folded = canonical.trim().to_lowercase();
        let reference = entries
            .iter()
            .find(|entry| entry.canonical.to_lowercase() == folded)
            .with_context(|| format!("Unknown correction term: {canonical}"))?;
        let spelling = reference.canonical.clone();
        let weight = reference.weight;
        let mut phones = Vec::new();
        for word in recognized.split_whitespace() {
            let normalized = word
                .trim_matches(|character: char| !character.is_alphanumeric() && character != '\'')
                .to_lowercase();
            phones.extend_from_slice(dictionary.get(&normalized).with_context(|| {
                format!("No query pronunciation for {word}; add it to the query dictionary")
            })?);
        }
        if phones.is_empty() || phones.len() > 63 {
            bail!("Correction pronunciation must contain 1 to 63 phonemes");
        }
        let reference = entries
            .iter()
            .filter(|entry| entry.canonical.to_lowercase() == folded && entry.source != 6)
            .min_by_key(|entry| alphabet::distance(&phones, &entry.phones, 252))
            .context("Corrections require an original pronunciation variant")?;
        align(&phones, &reference.phones, &mut counts);
        if seen.insert((folded, phones.clone())) {
            entries.push(Entry {
                canonical: spelling,
                phones,
                source: 6,
                weight,
            });
        }
    }
    let mut costs = alphabet::default_costs();
    for left in 0..39 {
        let total: u32 = counts[left].iter().sum();
        if total < 5 {
            continue;
        }
        for right in 0..39 {
            if left != right && counts[left][right] >= 3 {
                let probability = (counts[left][right] as f32 + 1.0) / (total as f32 + 40.0);
                costs[left][right] =
                    costs[left][right].min((-probability.ln()).round().clamp(1.0, 4.0) as u8);
            }
        }
    }
    alphabet::validate_costs(&costs)?;
    Ok(costs)
}

fn align(left: &[u8], right: &[u8], counts: &mut [[u32; 40]; 40]) {
    let costs = alphabet::default_costs();
    let mut matrix = [[0u16; 64]; 64];
    for (position, row) in matrix.iter_mut().enumerate() {
        row[0] = position as u16 * 3;
    }
    for (position, cost) in matrix[0].iter_mut().enumerate() {
        *cost = position as u16 * 3;
    }
    for (row, &left_phone) in left.iter().enumerate() {
        for (column, &right_phone) in right.iter().enumerate() {
            matrix[row + 1][column + 1] = (matrix[row][column + 1] + 3)
                .min(matrix[row + 1][column] + 3)
                .min(matrix[row][column] + costs[left_phone as usize][right_phone as usize] as u16);
        }
    }
    let (mut row, mut column) = (left.len(), right.len());
    while row > 0 || column > 0 {
        if row > 0
            && column > 0
            && matrix[row][column]
                == matrix[row - 1][column - 1]
                    + costs[left[row - 1] as usize][right[column - 1] as usize] as u16
        {
            counts[left[row - 1] as usize][right[column - 1] as usize] += 1;
            row -= 1;
            column -= 1;
        } else if row > 0 && matrix[row][column] == matrix[row - 1][column] + 3 {
            counts[left[row - 1] as usize][39] += 1;
            row -= 1;
        } else {
            counts[39][right[column - 1] as usize] += 1;
            column -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_reviewed_variants_once_and_rejects_unknown_terms() {
        let mut entries = vec![Entry {
            canonical: "git".into(),
            phones: alphabet::parse("G IH T").unwrap(),
            source: 0,
            weight: 1.0,
        }];
        let dictionary = BTreeMap::from([("get".into(), alphabet::parse("G EH T").unwrap())]);
        let costs = learn(&mut entries, &dictionary, &"get\tgit\n".repeat(100)).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].source, 6);
        assert!(costs[10][16] < alphabet::default_costs()[10][16]);
        assert!(learn(&mut entries, &dictionary, "get\tinvented").is_err());
        assert!(learn(&mut entries, &dictionary, "unknown\tgit").is_err());
    }
}
