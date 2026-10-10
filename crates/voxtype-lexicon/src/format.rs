use bytemuck::{Pod, Zeroable};

pub const SECTIONS: usize = 11;
pub const ARENA: usize = 0;
pub const VARIANTS: usize = 1;
pub const LENGTHS: usize = 2;
pub const GRAM_OFFSETS: usize = 3;
pub const GRAM_POSTINGS: usize = 4;
pub const KEYS: usize = 5;
pub const KEY_POSTINGS: usize = 6;
pub const TERMS: usize = 7;
pub const STRINGS: usize = 8;
pub const DICTIONARY: usize = 9;
pub const DICT_ARENA: usize = 10;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Section {
    pub offset: u64,
    pub length: u64,
    pub checksum: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Header {
    pub magic: [u8; 8],
    pub version: u32,
    pub variants: u32,
    pub terms: u32,
    pub reserved: u32,
    pub alphabet: u64,
    pub tokenizer: u64,
    pub sections: [Section; SECTIONS],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Variant {
    pub offset: u32,
    pub term: u32,
    pub length: u8,
    pub source: u8,
    pub padding: [u8; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct KeySlot {
    pub key: u64,
    pub offset: u32,
    pub length: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Term {
    pub offset: u32,
    pub length: u32,
    pub weight: u16,
    pub padding: [u8; 6],
}
