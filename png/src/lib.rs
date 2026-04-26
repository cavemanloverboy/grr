//! minimal PNG encoder. grayscale 8-bit only. zero dependencies :D
//!
//! uses DEFLATE with fixed Huffman codes (BTYPE=01) plus RLE-style
//! backreferences. real compression — sparse/repetitive images compress
//! to a fraction of raw size, which matters for black hole renders where
//! most of the image is background zeros.
//!
//! tradeoffs:
//! - fixed Huffman tree: skip the dynamic-tree complexity. suboptimal for
//!   non-uniform byte distributions, but simple and correct.
//! - RLE-only matching: scan backward for byte runs. misses general LZ77
//!   matches but catches the dominant case (zero runs, constant regions).
//! - one DEFLATE block per image: simplifies bookkeeping; no chunking.

use std::fs::File;
use std::io::{BufWriter, Write};

pub fn save_grayscale(pixels: &[u8], width: u32, height: u32, path: &str) -> std::io::Result<()> {
    assert_eq!(
        pixels.len(),
        (width * height) as usize,
        "pixel count mismatch"
    );

    let mut file = BufWriter::new(File::create(path)?);

    // png signature
    file.write_all(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A])?;

    // ihdr chunk
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(0); // colour type 0 = grayscale
    ihdr.push(0); // compression: deflate
    ihdr.push(0); // filter
    ihdr.push(0); // interlace: none
    write_chunk(&mut file, b"IHDR", &ihdr)?;

    // build the filtered scanline stream: each row prefixed by filter byte 0
    let mut raw = Vec::with_capacity((width * height + height) as usize);
    for y in 0..height as usize {
        raw.push(0);
        let row_start = y * width as usize;
        raw.extend_from_slice(&pixels[row_start..row_start + width as usize]);
    }

    // wrap raw in zlib + deflate (fixed huffman + rle)
    let zlib = wrap_zlib(&raw);
    write_chunk(&mut file, b"IDAT", &zlib)?;

    // iend
    write_chunk(&mut file, b"IEND", &[])?;

    Ok(())
}

fn write_chunk<W: Write>(w: &mut W, kind: &[u8; 4], data: &[u8]) -> std::io::Result<()> {
    w.write_all(&(data.len() as u32).to_be_bytes())?;
    w.write_all(kind)?;
    w.write_all(data)?;
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    w.write_all(&crc32(&crc_input).to_be_bytes())?;
    Ok(())
}

fn wrap_zlib(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2 + 16);
    // zlib header: cmf=0x78 (deflate, 32k window), flg=0x9c (default level)
    // (cmf*256 + flg) must be divisible by 31
    out.push(0x78);
    out.push(0x9c);

    deflate_fixed_huffman_rle(data, &mut out);

    // adler32 of uncompressed data
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn deflate_fixed_huffman_rle(data: &[u8], out: &mut Vec<u8>) {
    let mut bw = BitWriter::new(out);
    bw.write(1, 1); // BFINAL = 1 (single final block)
    bw.write(0b01, 2); // BTYPE = 01 (fixed huffman)

    let mut i = 0;
    while i < data.len() {
        // look for a run of repeated bytes starting at i. rle length must be
        // >= 3 to be worth encoding as a backreference (length-distance pair).
        // max length is 258, max distance is 32768 — but rle uses distance=1.
        let run_len = run_length_at(data, i);

        if run_len >= 3 {
            // encode as literal at i, then length-distance pair
            // (length, distance=1) for the remaining (run_len - 1) repeats.
            let literal = data[i];
            write_literal(&mut bw, literal);

            let mut remaining = run_len - 1;
            while remaining >= 3 {
                let chunk = remaining.min(258);
                write_length_distance(&mut bw, chunk, 1);
                remaining -= chunk;
            }
            // any leftover (1 or 2 bytes) goes as literals
            for _ in 0..remaining {
                write_literal(&mut bw, literal);
            }
            i += run_len;
        } else {
            write_literal(&mut bw, data[i]);
            i += 1;
        }
    }

    // end-of-block symbol = 256
    write_symbol(&mut bw, 256);
    bw.flush();
}

fn run_length_at(data: &[u8], start: usize) -> usize {
    let byte = data[start];
    let mut len = 1;
    while start + len < data.len() && data[start + len] == byte && len < 258 {
        len += 1;
    }
    len
}

/// returns (code, n_bits) for a literal/length symbol in [0, 287].
/// note: code is to be emitted msb-first into the bit stream.
fn fixed_huffman_lit_len(symbol: u16) -> (u32, u32) {
    match symbol {
        0..=143 => (symbol as u32 + 0b00110000, 8), // 0011_0000 .. 1011_1111
        144..=255 => (symbol as u32 - 144 + 0b110010000, 9), // 1_1001_0000 .. 1_1111_1111
        256..=279 => (symbol as u32 - 256 + 0b0000000, 7), // 000_0000 .. 001_0111
        280..=287 => (symbol as u32 - 280 + 0b11000000, 8), // 1100_0000 .. 1100_0111
        _ => unreachable!(),
    }
}

fn write_literal(bw: &mut BitWriter, byte: u8) {
    write_symbol(bw, byte as u16);
}

fn write_symbol(bw: &mut BitWriter, symbol: u16) {
    let (code, n_bits) = fixed_huffman_lit_len(symbol);
    bw.write_msb_first(code, n_bits);
}

/// length code table: symbol, extra_bits, base_length
/// from rfc 1951 section 3.2.5
const LENGTH_TABLE: [(u16, u32, u32); 29] = [
    (257, 0, 3),
    (258, 0, 4),
    (259, 0, 5),
    (260, 0, 6),
    (261, 0, 7),
    (262, 0, 8),
    (263, 0, 9),
    (264, 0, 10),
    (265, 1, 11),
    (266, 1, 13),
    (267, 1, 15),
    (268, 1, 17),
    (269, 2, 19),
    (270, 2, 23),
    (271, 2, 27),
    (272, 2, 31),
    (273, 3, 35),
    (274, 3, 43),
    (275, 3, 51),
    (276, 3, 59),
    (277, 4, 67),
    (278, 4, 83),
    (279, 4, 99),
    (280, 4, 115),
    (281, 5, 131),
    (282, 5, 163),
    (283, 5, 195),
    (284, 5, 227),
    (285, 0, 258),
];

/// distance code table: symbol, extra_bits, base_distance
/// from rfc 1951 section 3.2.5. distance codes 0-29 use 5 bits fixed.
const DISTANCE_TABLE: [(u16, u32, u32); 30] = [
    (0, 0, 1),
    (1, 0, 2),
    (2, 0, 3),
    (3, 0, 4),
    (4, 1, 5),
    (5, 1, 7),
    (6, 2, 9),
    (7, 2, 13),
    (8, 3, 17),
    (9, 3, 25),
    (10, 4, 33),
    (11, 4, 49),
    (12, 5, 65),
    (13, 5, 97),
    (14, 6, 129),
    (15, 6, 193),
    (16, 7, 257),
    (17, 7, 385),
    (18, 8, 513),
    (19, 8, 769),
    (20, 9, 1025),
    (21, 9, 1537),
    (22, 10, 2049),
    (23, 10, 3073),
    (24, 11, 4097),
    (25, 11, 6145),
    (26, 12, 8193),
    (27, 12, 12289),
    (28, 13, 16385),
    (29, 13, 24577),
];

fn write_length_distance(bw: &mut BitWriter, length: usize, distance: u32) {
    debug_assert!((3..=258).contains(&length));
    debug_assert!((1..=32768).contains(&distance));

    // length: find table entry, emit length symbol via fixed huffman, then extra bits
    let len = length as u32;
    let (sym, extra_bits, base) = *LENGTH_TABLE
        .iter()
        .rev()
        .find(|(_, eb, base)| *base <= len && len < *base + (1 << *eb))
        .expect("length out of range");
    write_symbol(bw, sym);
    if extra_bits > 0 {
        bw.write(len - base, extra_bits); // extra bits are lsb-first
    }

    // distance: 5-bit fixed code (msb-first), then extra bits (lsb-first)
    let (dsym, dextra, dbase) = *DISTANCE_TABLE
        .iter()
        .rev()
        .find(|(_, eb, base)| *base <= distance && distance < *base + (1 << *eb))
        .expect("distance out of range");
    bw.write_msb_first(dsym as u32, 5);
    if dextra > 0 {
        bw.write(distance - dbase, dextra);
    }
}

// deflate has a quirky bit packing convention:
// - bits within a byte are packed lsb-first (the first bit written goes to bit 0)
// - huffman codes are conceptually msb-first, but get packed into the stream
//   such that the msb of the code goes into the lower bit position
// in practice: we provide write() for lsb-first values (extra bits, block headers)
// and write_msb_first() for huffman codes.

struct BitWriter<'a> {
    buf: &'a mut Vec<u8>,
    bit_buffer: u64,
    bits_in_buffer: u32,
}

impl<'a> BitWriter<'a> {
    fn new(buf: &'a mut Vec<u8>) -> Self {
        BitWriter {
            buf,
            bit_buffer: 0,
            bits_in_buffer: 0,
        }
    }

    /// write `n_bits` of `value` lsb-first into the stream.
    fn write(&mut self, value: u32, n_bits: u32) {
        debug_assert!(n_bits <= 32);
        debug_assert!(n_bits == 32 || value < (1 << n_bits));
        self.bit_buffer |= (value as u64) << self.bits_in_buffer;
        self.bits_in_buffer += n_bits;
        while self.bits_in_buffer >= 8 {
            self.buf.push((self.bit_buffer & 0xFF) as u8);
            self.bit_buffer >>= 8;
            self.bits_in_buffer -= 8;
        }
    }

    /// write a huffman code msb-first: bit-reverse the code, then emit lsb-first.
    fn write_msb_first(&mut self, code: u32, n_bits: u32) {
        let mut reversed = 0u32;
        for i in 0..n_bits {
            if (code >> (n_bits - 1 - i)) & 1 == 1 {
                reversed |= 1 << i;
            }
        }
        self.write(reversed, n_bits);
    }

    fn flush(&mut self) {
        if self.bits_in_buffer > 0 {
            self.buf.push((self.bit_buffer & 0xFF) as u8);
            self.bit_buffer = 0;
            self.bits_in_buffer = 0;
        }
    }
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (!(crc & 1)).wrapping_add(1);
            crc = (crc >> 1) ^ (0xEDB88320 & mask);
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_values() {
        // crc32 of "123456789" with polynomial 0xedb88320 = 0xcbf43926
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn adler32_known_values() {
        // adler32 of "Wikipedia" = 0x11E60398
        assert_eq!(adler32(b"Wikipedia"), 0x11E60398);
    }

    #[test]
    fn write_tiny_image_roundtrips() {
        // smoke test: write a 4x4 grayscale gradient, file should exist and be
        // valid png signature.
        let pixels: Vec<u8> = (0..16).map(|i| (i * 16) as u8).collect();
        let path = "tests/grr_png_test.png";
        save_grayscale(&pixels, 4, 4, path).unwrap();
        let file_bytes = std::fs::read(path).unwrap();
        assert_eq!(
            &file_bytes[0..8],
            &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
        );
    }
}
