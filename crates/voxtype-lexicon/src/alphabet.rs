use anyhow::{bail, Result};

pub const PHONES: [&str; 39] = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH",
    "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH",
    "UW", "V", "W", "Y", "Z", "ZH",
];

pub const COARSE: [u8; 39] = [
    0, 0, 0, 0, 0, 0, 1, 8, 2, 5, 0, 11, 0, 4, 3, 14, 0, 0, 8, 3, 10, 9, 9, 9, 0, 0, 1, 11, 6, 7,
    2, 5, 0, 0, 4, 12, 13, 6, 7,
];

pub fn parse(text: &str) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    for token in text.split_whitespace() {
        let token = token.trim_end_matches(|character: char| character.is_ascii_digit());
        let Some(phone) = PHONES.iter().position(|phone| *phone == token) else {
            bail!("Unknown ARPAbet phone: {token}");
        };
        result.push(phone as u8);
    }
    if result.is_empty() || result.len() > 63 {
        bail!("Pronunciations must contain 1 to 63 phonemes");
    }
    Ok(result)
}

pub fn hash() -> u64 {
    xxhash_rust::xxh3::xxh3_64(PHONES.join(" ").as_bytes())
}

pub fn key(phones: &[u8]) -> u64 {
    let mut skeleton = [0; 64];
    let mut length = 0;
    for &phone in phones {
        let class = COARSE[phone as usize];
        if class != 0 && (length == 0 || skeleton[length - 1] != class) {
            skeleton[length] = class;
            length += 1;
        }
    }
    xxhash_rust::xxh3::xxh3_64(&skeleton[..length])
}

pub fn grams(phones: &[u8], output: &mut [u16; 65]) -> usize {
    let mut previous = 15;
    let mut current = 15;
    let mut length = 0;
    for next in phones
        .iter()
        .map(|phone| COARSE[*phone as usize])
        .chain([15, 15])
    {
        output[length] = ((previous as u16) << 8) | ((current as u16) << 4) | next as u16;
        previous = current;
        current = next;
        length += 1;
    }
    output[..length].sort_unstable();
    let mut unique = 0;
    for position in 0..length {
        if unique == 0 || output[position] != output[unique - 1] {
            output[unique] = output[position];
            unique += 1;
        }
    }
    unique
}

pub type Costs = [[u8; 40]; 40];

pub fn default_costs() -> Costs {
    let mut costs = [[3; 40]; 40];
    for left in 0..39 {
        for right in 0..39 {
            costs[left][right] = if left == right {
                0
            } else if COARSE[left] == 0 && COARSE[right] == 0 {
                2
            } else if COARSE[left] == COARSE[right] {
                1
            } else {
                4
            };
        }
    }
    costs[39][39] = 0;
    costs
}

pub fn validate_costs(costs: &Costs) -> Result<()> {
    for (left, row) in costs.iter().enumerate() {
        for (right, &cost) in row.iter().enumerate() {
            if (left == right && cost != 0)
                || (left != right && !(1..=4).contains(&cost))
                || ((left == 39 || right == 39) && left != right && cost != 3)
            {
                bail!("Invalid learned phoneme cost at {left},{right}");
            }
        }
    }
    Ok(())
}

pub fn distance(left: &[u8], right: &[u8], limit: u16) -> u16 {
    distance_with_costs(left, right, limit, &default_costs())
}

pub fn distance_with_costs(left: &[u8], right: &[u8], limit: u16, costs: &Costs) -> u16 {
    let infinity = u16::MAX / 2;
    let radius = (limit / 3) as usize;
    if left.len().abs_diff(right.len()) > radius {
        return limit.saturating_add(1);
    }
    let mut previous = [infinity; 64];
    let mut current = [infinity; 64];
    for (position, value) in previous.iter_mut().enumerate().take(right.len() + 1) {
        *value = position as u16 * 3;
    }
    for (row, &left_phone) in left.iter().enumerate() {
        current.fill(infinity);
        if row < radius {
            current[0] = (row as u16 + 1) * 3;
        }
        let mut minimum = current[0];
        let start = (row + 1).saturating_sub(radius).max(1) - 1;
        let end = (row + 1 + radius).min(right.len());
        for (column, &right_phone) in right.iter().enumerate().take(end).skip(start) {
            let substitution = costs[left_phone as usize][right_phone as usize] as u16;
            current[column + 1] = (previous[column + 1] + 3)
                .min(current[column] + 3)
                .min(previous[column] + substitution);
            minimum = minimum.min(current[column + 1]);
        }
        if minimum > limit {
            return limit.saturating_add(1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn reference(left: &[u8], right: &[u8]) -> u16 {
        let mut rows = vec![vec![0u16; right.len() + 1]; left.len() + 1];
        for (position, row) in rows.iter_mut().enumerate() {
            row[0] = position as u16 * 3;
        }
        for (position, value) in rows[0].iter_mut().enumerate() {
            *value = position as u16 * 3;
        }
        for (row, &left_phone) in left.iter().enumerate() {
            for (column, &right_phone) in right.iter().enumerate() {
                let cost = if left_phone == right_phone {
                    0
                } else if COARSE[left_phone as usize] == 0 && COARSE[right_phone as usize] == 0 {
                    2
                } else if COARSE[left_phone as usize] == COARSE[right_phone as usize] {
                    1
                } else {
                    4
                };
                rows[row + 1][column + 1] = (rows[row][column + 1] + 3)
                    .min(rows[row + 1][column] + 3)
                    .min(rows[row][column] + cost);
            }
        }
        rows[left.len()][right.len()]
    }

    proptest! {
        #[test]
        fn banded_distance_agrees_with_full_matrix(left in prop::collection::vec(0u8..39, 0..63), right in prop::collection::vec(0u8..39, 0..63), limit in 0u16..100) {
            let expected = reference(&left, &right);
            let actual = distance(&left, &right, limit);
            if expected <= limit { prop_assert_eq!(actual, expected); }
            else { prop_assert!(actual > limit); }
        }
    }
}
