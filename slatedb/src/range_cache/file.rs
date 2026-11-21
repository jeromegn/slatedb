//! Range cache files: the on-disk part of a cached range.
//!
//! A range cache file holds the live rows of one cached range at one sequence number. Rows are
//! sorted by key. A file never holds tombstones or rows with an expire time. The cache writes a
//! file once, into a temporary file that it renames into place, and then only reads it. The
//! file is deleted when the last reference to its [`RangeFile`] is dropped. The cache does not
//! sync files, because it can rebuild them from the database.
//!
//! Only the fence index (the first key and the offset of each page) is kept in memory. A
//! [`PageSource`] reads the pages. The memory-mapped source and the positional-read source
//! give the same results.
//!
//! File layout (all integers are little endian):
//!
//! ```text
//! [blob run][pad] page 0 [blob run][pad] page 1 ... page N-1
//! fence region:   for each page: u64 page offset | u32 key length | first key
//! trailer (64 B): magic u64 | version u32 | page size u32 | page count u64 | row count u64 |
//!                 fence offset u64 | fence length u64 | data bytes u64 | max seq u64
//! ```
//!
//! Each page is `page_size` bytes long and starts at a multiple of `page_size`:
//!
//! ```text
//! page:  u16 row count n | u16 hint skip s | n x u16 row offset | n x u32 key hint | rows |
//!        zero padding
//! hint:  bytes s..s+4 of the key of the row, big endian and zero padded (all keys of a page
//!        share their first s bytes)
//! row:   u8 flags | u64 seq | i64 create_ts | key ref | value ref
//! ref:   inline:      u32 length | bytes
//!        out of line: u64 file offset | u32 length  (when the flag bit is set)
//! ```
//!
//! A key longer than `page_size / 8`, and a value that makes its row longer than
//! `page_size / 4`, go out of line, into the blob run before the page that refers to them. So
//! each row fits in a page and the pages stay aligned.

use std::borrow::Cow;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::ops::{Bound, RangeBounds};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use slatedb_common::file_handle_cache::read_exact_at_offset;

use crate::bytes_range::BytesRange;
use crate::error::SlateDBError;
use crate::iter::RowEntryIterator;
use crate::types::{RowEntry, ValueDeletable};

/// The extension of finished range cache files.
pub(crate) const FILE_EXTENSION: &str = "rcf";
/// The extension of range cache files that are being written.
pub(crate) const TEMP_FILE_EXTENSION: &str = "rcf.tmp";

const MAGIC: u64 = u64::from_le_bytes(*b"SLATERCF");
const VERSION: u32 = 2;
const TRAILER_LEN: usize = 64;
/// Row count (u16) and the length of the key prefix that the hints skip (u16).
const PAGE_HEADER_LEN: usize = 4;
const ROW_OFFSET_LEN: usize = 2;
/// Four bytes of each key after the skipped prefix, big-endian: a search compares these, in
/// the page header, before it reads a row.
const ROW_HINT_LEN: usize = 4;
/// The longest prefix that hints skip. Writers keep this many bytes plus a hint of each key.
const MAX_HINT_SKIP: usize = 60;
/// Flags, seq and create_ts.
const ROW_FIXED_LEN: usize = 1 + 8 + 8;
const INLINE_REF_LEN: usize = 4;
const OUT_OF_LINE_REF_LEN: usize = 12;

const FLAG_KEY_OUT_OF_LINE: u8 = 1;
const FLAG_VALUE_OUT_OF_LINE: u8 = 2;
/// The row is a merge operand with no base value below it.
const FLAG_MERGE: u8 = 4;
/// The row has a create timestamp.
const FLAG_CREATE_TS: u8 = 8;

pub(crate) const MIN_PAGE_SIZE: usize = 512;
pub(crate) const MAX_PAGE_SIZE: usize = 65536;

/// Page sizes must be powers of two in `512..=65536`, because row offsets are `u16`.
pub(crate) fn valid_page_size(page_size: usize) -> bool {
    page_size.is_power_of_two() && (MIN_PAGE_SIZE..=MAX_PAGE_SIZE).contains(&page_size)
}

fn u16_at(b: &[u8], off: usize) -> usize {
    u16::from_le_bytes([b[off], b[off + 1]]) as usize
}

fn u32_at(b: &[u8], off: usize) -> usize {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]) as usize
}

fn u64_at(b: &[u8], off: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(buf)
}

/// The first eight bytes of `bytes` as a big-endian number, zero-padded. Two keys whose
/// numbers differ compare like their numbers.
fn prefix64(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let n = bytes.len().min(8);
    buf[..n].copy_from_slice(&bytes[..n]);
    u64::from_be_bytes(buf)
}

/// The first four bytes of `bytes` as a big-endian number, zero-padded. Like `prefix64`.
fn prefix32(bytes: &[u8]) -> u32 {
    let mut buf = [0u8; 4];
    let n = bytes.len().min(4);
    buf[..n].copy_from_slice(&bytes[..n]);
    u32::from_be_bytes(buf)
}

fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn corrupt(msg: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("range cache file: {msg}"),
    )
}

/// How the range cache reads its files.
///
/// Both sources return the same results.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum RangeCachePageSource {
    /// Map each file into memory. Reads make no system calls when the pages are in the page
    /// cache.
    #[default]
    Mmap,
    /// Read the files with positional reads (`pread`).
    PositionalRead,
}

/// Reads the bytes of one immutable file.
pub(crate) trait PageSource: Send + Sync {
    /// Returns the whole file if the source maps it into memory.
    fn mapped(&self) -> Option<&[u8]>;
    /// Fills `buf` with the bytes at `offset`.
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()>;
}

struct MmapPageSource {
    map: memmap2::Mmap,
}

impl PageSource for MmapPageSource {
    fn mapped(&self) -> Option<&[u8]> {
        Some(&self.map)
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        buf.copy_from_slice(slice_at(&self.map, offset, buf.len())?);
        Ok(())
    }
}

struct PositionalReadPageSource {
    file: File,
}

impl PageSource for PositionalReadPageSource {
    fn mapped(&self) -> Option<&[u8]> {
        None
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        read_exact_at_offset(&self.file, buf, offset)
    }
}

fn slice_at(bytes: &[u8], offset: u64, len: usize) -> io::Result<&[u8]> {
    usize::try_from(offset)
        .ok()
        .and_then(|start| bytes.get(start..start.checked_add(len)?))
        .ok_or_else(|| corrupt("read past the end of the file"))
}

/// Reads `len` bytes at `offset`. A memory-mapped source returns them without a copy.
fn read_bytes(source: &dyn PageSource, offset: u64, len: usize) -> io::Result<Cow<'_, [u8]>> {
    match source.mapped() {
        Some(map) => slice_at(map, offset, len).map(Cow::Borrowed),
        None => {
            let mut buf = vec![0u8; len];
            source.read_exact_at(&mut buf, offset)?;
            Ok(Cow::Owned(buf))
        }
    }
}

/// Copies a key and a value into one allocation.
pub(super) fn copy_key_value(key: &[u8], value: &[u8]) -> (Bytes, Bytes) {
    let mut buf = BytesMut::with_capacity(key.len() + value.len());
    buf.extend_from_slice(key);
    buf.extend_from_slice(value);
    let buf = buf.freeze();
    (buf.slice(..key.len()), buf.slice(key.len()..))
}

/// Where the bytes of a key or a value are.
enum Ref {
    /// In the page, at this range.
    Inline(std::ops::Range<usize>),
    /// Out of line, in the file.
    OutOfLine { offset: u64, len: usize },
}

/// The decoded header and references of one row.
struct RawRow {
    flags: u8,
    seq: u64,
    create_ts: i64,
    key: Ref,
    value: Ref,
}

fn parse_ref(page: &[u8], at: usize, out_of_line: bool) -> io::Result<(Ref, usize)> {
    if out_of_line {
        if at + OUT_OF_LINE_REF_LEN > page.len() {
            return Err(corrupt("truncated reference"));
        }
        let offset = u64_at(page, at);
        let len = u32_at(page, at + 8);
        Ok((Ref::OutOfLine { offset, len }, at + OUT_OF_LINE_REF_LEN))
    } else {
        if at + INLINE_REF_LEN > page.len() {
            return Err(corrupt("truncated reference"));
        }
        let len = u32_at(page, at);
        let start = at + INLINE_REF_LEN;
        if start + len > page.len() {
            return Err(corrupt("truncated inline bytes"));
        }
        Ok((Ref::Inline(start..start + len), start + len))
    }
}

/// Decodes row `index` of `page`.
fn parse_row(page: &[u8], index: usize) -> io::Result<RawRow> {
    parse_row_at(page, u16_at(page, PAGE_HEADER_LEN + ROW_OFFSET_LEN * index))
}

/// Decodes the row at offset `off` of `bytes`, a page or a part of one. The references of the
/// row are relative to `bytes`.
fn parse_row_at(page: &[u8], off: usize) -> io::Result<RawRow> {
    if off + ROW_FIXED_LEN > page.len() {
        return Err(corrupt("bad row offset"));
    }
    let flags = page[off];
    let seq = u64_at(page, off + 1);
    let create_ts = u64_at(page, off + 9) as i64;
    let (key, next) = parse_ref(page, off + ROW_FIXED_LEN, flags & FLAG_KEY_OUT_OF_LINE != 0)?;
    let (value, _) = parse_ref(page, next, flags & FLAG_VALUE_OUT_OF_LINE != 0)?;
    Ok(RawRow {
        flags,
        seq,
        create_ts,
        key,
        value,
    })
}

fn open_page_source(
    file: File,
    len: u64,
    kind: RangeCachePageSource,
) -> io::Result<Box<dyn PageSource>> {
    match kind {
        RangeCachePageSource::Mmap => {
            // SAFETY: the range cache owns its directory. It writes each file once, renames
            // it into place before it maps it, and never changes it after that. It only
            // unlinks files, and an unlinked file stays valid while it is mapped.
            #[allow(unsafe_code)]
            let map = unsafe { memmap2::Mmap::map(&file)? };
            if map.len() as u64 != len {
                return Err(corrupt("file length changed while it was opened"));
            }
            Ok(Box::new(MmapPageSource { map }))
        }
        RangeCachePageSource::PositionalRead => Ok(Box::new(PositionalReadPageSource { file })),
    }
}

/// Writes one range cache file from rows in strictly ascending key order.
///
/// The writer writes `<name>.rcf.tmp` and [`RangeFileWriter::finish`] renames it to
/// `<name>.rcf`. A writer that is dropped before it finishes removes its temporary file.
pub(crate) struct RangeFileWriter {
    tmp_path: PathBuf,
    final_path: PathBuf,
    out: Option<BufWriter<File>>,
    page_size: usize,
    /// Bytes written to the file so far.
    pos: u64,
    /// Rows of the open page, without the page header and the row offsets.
    body: Vec<u8>,
    /// Offsets of the rows in `body`.
    offsets: Vec<usize>,
    /// The first `MAX_HINT_SKIP + ROW_HINT_LEN` bytes of each key of the open page.
    hint_keys: Vec<u8>,
    /// `hint_keys` ends of each key.
    hint_key_ends: Vec<usize>,
    first_key: Vec<u8>,
    fences: Vec<u8>,
    page_buf: Vec<u8>,
    pages: u64,
    rows: u64,
    data_bytes: u64,
    max_seq: u64,
    prev_key: Option<Vec<u8>>,
}

impl RangeFileWriter {
    /// Starts `dir/<name>.rcf.tmp`.
    pub(crate) fn create(dir: &Path, name: &str, page_size: usize) -> io::Result<Self> {
        if !valid_page_size(page_size) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("page size {page_size} is not a power of two in 512..=65536"),
            ));
        }
        let tmp_path = dir.join(format!("{name}.{TEMP_FILE_EXTENSION}"));
        let final_path = dir.join(format!("{name}.{FILE_EXTENSION}"));
        let file = File::create(&tmp_path)?;
        Ok(Self {
            tmp_path,
            final_path,
            out: Some(BufWriter::with_capacity(256 * 1024, file)),
            page_size,
            pos: 0,
            body: Vec::with_capacity(page_size),
            offsets: Vec::new(),
            hint_keys: Vec::new(),
            hint_key_ends: Vec::new(),
            first_key: Vec::new(),
            fences: Vec::new(),
            page_buf: vec![0; page_size],
            pages: 0,
            rows: 0,
            data_bytes: 0,
            max_seq: 0,
            prev_key: None,
        })
    }

    /// Returns about the size of the file if it were finished now.
    pub(crate) fn estimated_len(&self) -> u64 {
        let open_page = if self.offsets.is_empty() {
            0
        } else {
            self.page_size as u64
        };
        self.pos + open_page + self.fences.len() as u64 + TRAILER_LEN as u64
    }

    pub(crate) fn rows(&self) -> u64 {
        self.rows
    }

    fn write_raw(&mut self, bytes: &[u8]) -> io::Result<()> {
        match self.out.as_mut() {
            Some(out) => out.write_all(bytes)?,
            None => return Err(io::Error::other("range cache file writer is finished")),
        }
        self.pos += bytes.len() as u64;
        Ok(())
    }

    fn pad_to_page(&mut self) -> io::Result<()> {
        const ZEROS: [u8; 4096] = [0; 4096];
        let page_size = self.page_size as u64;
        let mut pad = (page_size - self.pos % page_size) % page_size;
        while pad > 0 {
            let n = pad.min(ZEROS.len() as u64) as usize;
            self.write_raw(&ZEROS[..n])?;
            pad -= n as u64;
        }
        Ok(())
    }

    /// Appends one row. The row must be a value or a merge operand, must not expire, and its
    /// key must be greater than the key of the previous row.
    pub(crate) fn push(&mut self, row: &RowEntry) -> io::Result<()> {
        let (value, merge) = match &row.value {
            ValueDeletable::Value(v) => (v, false),
            ValueDeletable::Merge(v) => (v, true),
            ValueDeletable::Tombstone => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "range cache files do not hold tombstones",
                ))
            }
        };
        if row.expire_ts.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "range cache files do not hold rows with an expire time",
            ));
        }
        let key = row.key.as_ref();
        if let Some(prev) = &self.prev_key {
            if key <= prev.as_slice() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "range cache file rows must be in strictly ascending key order",
                ));
            }
        }
        if key.len() > u32::MAX as usize || value.len() > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "keys and values are limited to u32::MAX bytes",
            ));
        }
        let page_size = self.page_size;
        let key_ool = key.len() > page_size / 8;
        let key_ref_len = if key_ool {
            OUT_OF_LINE_REF_LEN
        } else {
            INLINE_REF_LEN + key.len()
        };
        let value_ool = ROW_FIXED_LEN + key_ref_len + INLINE_REF_LEN + value.len() > page_size / 4;
        let value_ref_len = if value_ool {
            OUT_OF_LINE_REF_LEN
        } else {
            INLINE_REF_LEN + value.len()
        };
        let row_len = ROW_FIXED_LEN + key_ref_len + value_ref_len;
        if !self.offsets.is_empty()
            && PAGE_HEADER_LEN
                + (ROW_OFFSET_LEN + ROW_HINT_LEN) * (self.offsets.len() + 1)
                + self.body.len()
                + row_len
                > page_size
        {
            self.flush_page()?;
        }
        // Blobs go into the run before the page that holds this row.
        let key_off = if key_ool {
            let off = self.pos;
            self.write_raw(key)?;
            Some(off)
        } else {
            None
        };
        let value_off = if value_ool {
            let off = self.pos;
            self.write_raw(value)?;
            Some(off)
        } else {
            None
        };
        if self.offsets.is_empty() {
            self.first_key.clear();
            self.first_key.extend_from_slice(key);
        }
        self.offsets.push(self.body.len());
        self.hint_keys
            .extend_from_slice(&key[..key.len().min(MAX_HINT_SKIP + ROW_HINT_LEN)]);
        self.hint_key_ends.push(self.hint_keys.len());
        let mut flags = 0;
        if key_ool {
            flags |= FLAG_KEY_OUT_OF_LINE;
        }
        if value_ool {
            flags |= FLAG_VALUE_OUT_OF_LINE;
        }
        if merge {
            flags |= FLAG_MERGE;
        }
        if row.create_ts.is_some() {
            flags |= FLAG_CREATE_TS;
        }
        self.body.push(flags);
        self.body.extend_from_slice(&row.seq.to_le_bytes());
        self.body
            .extend_from_slice(&row.create_ts.unwrap_or(0).to_le_bytes());
        for (bytes, off) in [(key, key_off), (value.as_ref(), value_off)] {
            match off {
                Some(off) => {
                    self.body.extend_from_slice(&off.to_le_bytes());
                    self.body
                        .extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                }
                None => {
                    self.body
                        .extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                    self.body.extend_from_slice(bytes);
                }
            }
        }
        self.rows += 1;
        self.data_bytes += (key.len() + value.len()) as u64;
        self.max_seq = self.max_seq.max(row.seq);
        match &mut self.prev_key {
            Some(prev) => {
                prev.clear();
                prev.extend_from_slice(key);
            }
            None => self.prev_key = Some(key.to_vec()),
        }
        Ok(())
    }

    fn flush_page(&mut self) -> io::Result<()> {
        self.pad_to_page()?;
        let page_off = self.pos;
        self.fences.extend_from_slice(&page_off.to_le_bytes());
        self.fences
            .extend_from_slice(&(self.first_key.len() as u32).to_le_bytes());
        self.fences.extend_from_slice(&self.first_key);

        let mut page = std::mem::take(&mut self.page_buf);
        page.fill(0);
        let n = self.offsets.len();
        // Keys of a page are sorted, so the first and the last share the prefix that all share.
        let last_key = self.prev_key.as_deref().unwrap_or_default();
        let skip = common_prefix_len(&self.first_key, last_key).min(MAX_HINT_SKIP);
        page[..2].copy_from_slice(&(n as u16).to_le_bytes());
        page[2..4].copy_from_slice(&(skip as u16).to_le_bytes());
        let hints = PAGE_HEADER_LEN + ROW_OFFSET_LEN * n;
        let base = hints + ROW_HINT_LEN * n;
        let mut key_start = 0;
        for (i, off) in self.offsets.iter().enumerate() {
            let at = PAGE_HEADER_LEN + ROW_OFFSET_LEN * i;
            page[at..at + 2].copy_from_slice(&((base + off) as u16).to_le_bytes());
            let key = &self.hint_keys[key_start..self.hint_key_ends[i]];
            key_start = self.hint_key_ends[i];
            let hint = prefix32(key.get(skip..).unwrap_or_default());
            let at = hints + ROW_HINT_LEN * i;
            page[at..at + ROW_HINT_LEN].copy_from_slice(&hint.to_be_bytes());
        }
        page[base..base + self.body.len()].copy_from_slice(&self.body);
        let written = self.write_raw(&page);
        self.page_buf = page;
        written?;
        self.body.clear();
        self.offsets.clear();
        self.hint_keys.clear();
        self.hint_key_ends.clear();
        self.pages += 1;
        Ok(())
    }

    /// Writes the last page, the fence region and the trailer, and renames the file into
    /// place. Returns the final path.
    pub(crate) fn finish(mut self) -> io::Result<PathBuf> {
        if !self.offsets.is_empty() {
            self.flush_page()?;
        }
        let fence_off = self.pos;
        let fences = std::mem::take(&mut self.fences);
        self.write_raw(&fences)?;
        let mut trailer = Vec::with_capacity(TRAILER_LEN);
        trailer.extend_from_slice(&MAGIC.to_le_bytes());
        trailer.extend_from_slice(&VERSION.to_le_bytes());
        trailer.extend_from_slice(&(self.page_size as u32).to_le_bytes());
        for x in [
            self.pages,
            self.rows,
            fence_off,
            fences.len() as u64,
            self.data_bytes,
            self.max_seq,
        ] {
            trailer.extend_from_slice(&x.to_le_bytes());
        }
        debug_assert_eq!(trailer.len(), TRAILER_LEN);
        self.write_raw(&trailer)?;
        let out = self
            .out
            .take()
            .ok_or_else(|| io::Error::other("range cache file writer is finished"))?;
        out.into_inner().map_err(|e| e.into_error())?;
        fs::rename(&self.tmp_path, &self.final_path)?;
        Ok(self.final_path.clone())
    }
}

impl Drop for RangeFileWriter {
    fn drop(&mut self) {
        if self.out.take().is_some() {
            let _ = fs::remove_file(&self.tmp_path);
        }
    }
}

/// An open, immutable range cache file. Dropping the last reference deletes the file.
pub(crate) struct RangeFile {
    path: PathBuf,
    source: Box<dyn PageSource>,
    page_size: usize,
    page_offsets: Vec<u64>,
    /// The first key of each page, concatenated.
    fence_keys: Vec<u8>,
    /// `fence_key_offsets[i]..fence_key_offsets[i + 1]` is the first key of page `i`.
    fence_key_offsets: Vec<usize>,
    /// The length of the prefix that all fence keys share.
    fence_lcp: usize,
    /// For each page, eight bytes of its fence key after the shared prefix, big-endian and
    /// zero-padded. A search compares these first, in one contiguous array, and compares
    /// whole keys only on a tie.
    fence_prefixes: Vec<u64>,
    rows: u64,
    data_bytes: u64,
    max_seq: u64,
    file_len: u64,
    /// The end of the data pages and blob runs, which is the start of the fence region.
    data_end: u64,
    /// The bytes of all open range cache files of one cache.
    disk_bytes: Arc<AtomicU64>,
}

impl std::fmt::Debug for RangeFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RangeFile")
            .field("path", &self.path)
            .field("pages", &self.page_offsets.len())
            .field("rows", &self.rows)
            .finish()
    }
}

impl RangeFile {
    /// Opens a finished file. After this call, the `RangeFile` owns the file and deletes it
    /// when it is dropped, also when the open fails.
    pub(crate) fn open(
        path: PathBuf,
        page_source: RangeCachePageSource,
        disk_bytes: Arc<AtomicU64>,
    ) -> io::Result<Self> {
        match Self::open_inner(&path, page_source, disk_bytes) {
            Ok(file) => Ok(file),
            Err(e) => {
                let _ = fs::remove_file(&path);
                Err(e)
            }
        }
    }

    fn open_inner(
        path: &Path,
        page_source: RangeCachePageSource,
        disk_bytes: Arc<AtomicU64>,
    ) -> io::Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len < TRAILER_LEN as u64 {
            return Err(corrupt("file is shorter than its trailer"));
        }
        let source = open_page_source(file, file_len, page_source)?;
        let trailer = read_bytes(source.as_ref(), file_len - TRAILER_LEN as u64, TRAILER_LEN)?;
        let t = trailer.as_ref();
        if u64_at(t, 0) != MAGIC {
            return Err(corrupt("bad magic"));
        }
        if u32_at(t, 8) as u32 != VERSION {
            return Err(corrupt("unsupported version"));
        }
        let page_size = u32_at(t, 12);
        let pages = u64_at(t, 16) as usize;
        let rows = u64_at(t, 24);
        let fence_off = u64_at(t, 32);
        let fence_len = u64_at(t, 40);
        let data_bytes = u64_at(t, 48);
        let max_seq = u64_at(t, 56);
        if !valid_page_size(page_size) || fence_off + fence_len + TRAILER_LEN as u64 != file_len {
            return Err(corrupt("inconsistent trailer"));
        }
        let fence_region = read_bytes(source.as_ref(), fence_off, fence_len as usize)?;
        let raw = fence_region.as_ref();
        let mut page_offsets = Vec::with_capacity(pages);
        let mut fence_keys = Vec::new();
        let mut fence_key_offsets = Vec::with_capacity(pages + 1);
        fence_key_offsets.push(0);
        let mut at = 0;
        for _ in 0..pages {
            if at + 12 > raw.len() {
                return Err(corrupt("truncated fence region"));
            }
            let off = u64_at(raw, at);
            let key_len = u32_at(raw, at + 8);
            if at + 12 + key_len > raw.len()
                || !off.is_multiple_of(page_size as u64)
                || off + page_size as u64 > fence_off
            {
                return Err(corrupt("bad fence entry"));
            }
            page_offsets.push(off);
            fence_keys.extend_from_slice(&raw[at + 12..at + 12 + key_len]);
            fence_key_offsets.push(fence_keys.len());
            at += 12 + key_len;
        }
        if at != raw.len() {
            return Err(corrupt("fence region length mismatch"));
        }
        let fence = |i: usize| &fence_keys[fence_key_offsets[i]..fence_key_offsets[i + 1]];
        // Fences are sorted, so the first and the last share the prefix that all share.
        let fence_lcp = match pages {
            0 => 0,
            _ => common_prefix_len(fence(0), fence(pages - 1)),
        };
        let fence_prefixes = (0..pages)
            .map(|i| prefix64(fence(i).get(fence_lcp..).unwrap_or_default()))
            .collect();
        drop(fence_region);
        drop(trailer);
        disk_bytes.fetch_add(file_len, Ordering::Relaxed);
        Ok(Self {
            path: path.to_path_buf(),
            source,
            page_size,
            page_offsets,
            fence_keys,
            fence_key_offsets,
            fence_lcp,
            fence_prefixes,
            rows,
            data_bytes,
            max_seq,
            file_len,
            data_end: fence_off,
            disk_bytes,
        })
    }

    pub(crate) fn rows(&self) -> u64 {
        self.rows
    }

    /// The size of the file on disk.
    pub(crate) fn file_len(&self) -> u64 {
        self.file_len
    }

    /// The sum of the key and value lengths of all rows.
    #[allow(dead_code)]
    pub(crate) fn data_bytes(&self) -> u64 {
        self.data_bytes
    }

    /// The greatest sequence number of a row in the file.
    #[allow(dead_code)]
    pub(crate) fn max_seq(&self) -> u64 {
        self.max_seq
    }

    fn page_count(&self) -> usize {
        self.page_offsets.len()
    }

    fn fence(&self, page: usize) -> &[u8] {
        &self.fence_keys[self.fence_key_offsets[page]..self.fence_key_offsets[page + 1]]
    }

    /// The number of pages whose first key is `<= key`.
    fn pages_le(&self, key: &[u8]) -> usize {
        let pages = self.page_count();
        if pages == 0 {
            return 0;
        }
        // Compare with the shared prefix first: a key outside it is before or after all pages.
        let lcp = self.fence_lcp;
        let shared = &self.fence(0)[..lcp];
        let head = &key[..lcp.min(key.len())];
        match head.cmp(&shared[..head.len()]) {
            std::cmp::Ordering::Less => return 0,
            std::cmp::Ordering::Greater => return pages,
            std::cmp::Ordering::Equal if key.len() < lcp => return 0,
            std::cmp::Ordering::Equal => {}
        }
        // Padded prefixes that differ order their keys the same way. Equal ones need the keys.
        let target = prefix64(&key[lcp..]);
        let (mut lo, mut hi) = (0, pages);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let prefix = self.fence_prefixes[mid];
            let le = prefix < target || (prefix == target && self.fence(mid) <= key);
            if le {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn read(&self, offset: u64, len: usize) -> io::Result<Cow<'_, [u8]>> {
        read_bytes(self.source.as_ref(), offset, len)
    }

    fn page(&self, page: usize) -> io::Result<Cow<'_, [u8]>> {
        self.read(self.page_offsets[page], self.page_size)
    }

    fn row_count(page: &[u8]) -> usize {
        u16_at(page, 0)
    }

    /// Returns the bytes of a key or a value of a row in `page`.
    fn resolve<'a>(&'a self, page: &'a [u8], r: &Ref) -> io::Result<Cow<'a, [u8]>> {
        match r {
            Ref::Inline(range) => Ok(Cow::Borrowed(&page[range.clone()])),
            Ref::OutOfLine { offset, len } => self.read(*offset, *len),
        }
    }

    fn key_at<'a>(&'a self, page: &'a [u8], index: usize) -> io::Result<Cow<'a, [u8]>> {
        let raw = parse_row(page, index)?;
        self.resolve(page, &raw.key)
    }

    /// Copies a row out of `page`, with its key and value in one allocation.
    fn decode(&self, page: &[u8], raw: &RawRow) -> io::Result<RowEntry> {
        let key = self.resolve(page, &raw.key)?;
        let value = self.resolve(page, &raw.value)?;
        let (key, value) = copy_key_value(&key, &value);
        Ok(RowEntry {
            key,
            value: if raw.flags & FLAG_MERGE != 0 {
                ValueDeletable::Merge(value)
            } else {
                ValueDeletable::Value(value)
            },
            seq: raw.seq,
            create_ts: (raw.flags & FLAG_CREATE_TS != 0).then_some(raw.create_ts),
            expire_ts: None,
        })
    }

    /// The index of the first row in `page` whose key is `>= key` (`> key` when `strict`).
    /// `first` is the first key of the page, its fence.
    fn search(&self, page: &[u8], key: &[u8], strict: bool, first: &[u8]) -> io::Result<usize> {
        let rows = Self::row_count(page);
        let skip = u16_at(page, 2);
        let hints = PAGE_HEADER_LEN + ROW_OFFSET_LEN * rows;
        if hints + ROW_HINT_LEN * rows > page.len() || skip > first.len() {
            return Err(corrupt("bad page header"));
        }
        // All keys of the page start with `first[..skip]`. A key that does not shares no
        // order with the hints: compare whole keys.
        let target = key
            .get(..skip)
            .filter(|head| *head == &first[..skip])
            .map(|_| prefix32(&key[skip..]));
        let (mut lo, mut hi) = (0, rows);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let at = hints + ROW_HINT_LEN * mid;
            let hint = u32::from_be_bytes([page[at], page[at + 1], page[at + 2], page[at + 3]]);
            // Hints that differ order their keys the same way. Equal ones need the keys.
            let less = match target {
                Some(t) if hint < t => true,
                Some(t) if hint > t => false,
                _ => {
                    let k = self.key_at(page, mid)?;
                    k.as_ref() < key || (strict && k.as_ref() == key)
                }
            };
            if less {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        Ok(lo)
    }

    /// Returns the row for `key`, if the file holds one.
    pub(crate) fn get(&self, key: &[u8]) -> io::Result<Option<RowEntry>> {
        let pages = self.pages_le(key);
        if pages == 0 {
            return Ok(None);
        }
        let page = self.page(pages - 1)?;
        let index = self.search(&page, key, false, self.fence(pages - 1))?;
        if index < Self::row_count(&page) {
            let raw = parse_row(&page, index)?;
            if self.resolve(&page, &raw.key)?.as_ref() == key {
                return self.decode(&page, &raw).map(Some);
            }
        }
        Ok(None)
    }

    /// The position of the first row at or after `start`, as (page, index), reading pages
    /// through `reader`. `(pages, 0)` is the end.
    fn locate(&self, reader: &mut PageReader, start: Bound<&[u8]>) -> io::Result<(usize, usize)> {
        let (key, strict) = match start {
            Bound::Unbounded => return Ok((0, 0)),
            Bound::Included(k) => (k, false),
            Bound::Excluded(k) => (k, true),
        };
        let pages = self.pages_le(key);
        if pages == 0 {
            return Ok((0, 0));
        }
        let page = reader.page(self, pages - 1)?;
        let index = self.search(page, key, strict, self.fence(pages - 1))?;
        if index < Self::row_count(page) {
            Ok((pages - 1, index))
        } else {
            Ok((pages, 0))
        }
    }
}

impl Drop for RangeFile {
    fn drop(&mut self) {
        self.disk_bytes.fetch_sub(self.file_len, Ordering::Relaxed);
        let _ = fs::remove_file(&self.path);
    }
}

/// The largest read of a scan with positional reads.
const MAX_READ_AHEAD_BYTES: usize = 64 * 1024;

/// Gives a scan the bytes of whole pages.
///
/// With a memory map, it returns slices of the map. With positional reads, it reads whole pages
/// into a buffer, and doubles the read for each refill up to [`MAX_READ_AHEAD_BYTES`]. So a
/// short scan reads one page, and a scan of 100 rows of 4 KiB pages makes about three reads.
pub(crate) struct PageReader {
    buf: Vec<u8>,
    offset: u64,
    next_len: usize,
}

impl PageReader {
    pub(crate) fn new() -> Self {
        Self {
            buf: Vec::new(),
            offset: 0,
            next_len: 0,
        }
    }

    fn page<'a>(&'a mut self, file: &'a RangeFile, page: usize) -> io::Result<&'a [u8]> {
        let offset = file.page_offsets[page];
        let len = file.page_size;
        if let Some(map) = file.source.mapped() {
            return slice_at(map, offset, len);
        }
        let buffered =
            offset >= self.offset && offset + len as u64 <= self.offset + self.buf.len() as u64;
        if !buffered {
            let read_len = self.next_len.max(len) as u64;
            let read_len = read_len.min(file.data_end - offset).max(len as u64) as usize;
            self.buf.resize(read_len, 0);
            file.source.read_exact_at(&mut self.buf, offset)?;
            self.offset = offset;
            self.next_len = (read_len * 2).min(MAX_READ_AHEAD_BYTES);
        }
        let start = (offset - self.offset) as usize;
        Ok(&self.buf[start..start + len])
    }
}

/// A forward cursor over the rows of a [`RangeFile`] in a key range. It holds no reference to
/// the file: each call that reads takes the file, and the caller keeps it open while the cursor
/// lives. A scan passes the file of the layers that it pinned, so it changes no reference count
/// that other reads of the file share.
///
/// When the cursor enters a page, it copies the page, from the first row that it reads to the end,
/// into one reference-counted buffer. It reads keys from that buffer, and returns rows as slices
/// of it, so it makes one allocation for each page, not for each row. A row that the caller keeps
/// keeps that buffer (at most a page) in memory, as a value from an SST keeps its block. It never
/// keeps the file or the mapping. Out-of-line keys and values are copied on their own.
pub(crate) struct RangeFileCursor {
    /// The address of the file, to check that each call passes the same one.
    #[cfg(debug_assertions)]
    file: usize,
    end: Bound<Bytes>,
    page: usize,
    index: usize,
    reader: PageReader,
    /// The page that `page_bytes` holds.
    loaded_page: Option<usize>,
    /// A copy of `loaded_page` from the first row that the cursor reads to the end of the
    /// page. Rows are slices of it.
    page_bytes: Bytes,
    /// The offset in the page of the first byte of `page_bytes`.
    page_base: usize,
    /// The row offsets of `loaded_page`.
    row_offsets: Vec<u16>,
    /// The decoded current row, or `None` past the last row in the range.
    current: Option<RawRow>,
    /// The key of the current row, if the row stores it out of line.
    out_of_line_key: Option<Bytes>,
}

impl RangeFileCursor {
    pub(crate) fn new(file: &RangeFile, range: &BytesRange) -> io::Result<Self> {
        let start = range.start_bound().map(|b| b.as_ref());
        Self::with_bounds(file, start, range.end_bound().cloned())
    }

    /// Returns a cursor over the rows of `file` from `start` to `end`.
    pub(crate) fn with_bounds(
        file: &RangeFile,
        start: Bound<&[u8]>,
        end: Bound<Bytes>,
    ) -> io::Result<Self> {
        let mut reader = PageReader::new();
        let (page, index) = file.locate(&mut reader, start)?;
        let mut cursor = Self {
            #[cfg(debug_assertions)]
            file: std::ptr::from_ref(file).addr(),
            end,
            page,
            index,
            reader,
            loaded_page: None,
            page_bytes: Bytes::new(),
            page_base: 0,
            row_offsets: Vec::new(),
            current: None,
            out_of_line_key: None,
        };
        cursor.normalize(file)?;
        Ok(cursor)
    }

    fn check(&self, file: &RangeFile) {
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            self.file,
            std::ptr::from_ref(file).addr(),
            "a range file cursor read through another file"
        );
        let _ = file;
    }

    /// Moves to the next row if the position is past the end of a page, decodes the row, and
    /// checks the end bound.
    fn normalize(&mut self, file: &RangeFile) -> io::Result<()> {
        self.check(file);
        self.current = None;
        self.out_of_line_key = None;
        loop {
            if self.page >= file.page_count() {
                return Ok(());
            }
            if self.loaded_page != Some(self.page) {
                let page = self.reader.page(file, self.page)?;
                let rows = RangeFile::row_count(page);
                if PAGE_HEADER_LEN + (ROW_OFFSET_LEN + ROW_HINT_LEN) * rows > page.len() {
                    return Err(corrupt("bad page header"));
                }
                if self.index >= rows {
                    self.page += 1;
                    self.index = 0;
                    continue;
                }
                self.row_offsets.clear();
                self.row_offsets.extend(
                    (0..rows).map(|i| u16_at(page, PAGE_HEADER_LEN + ROW_OFFSET_LEN * i) as u16),
                );
                // Copy only from the first row that the cursor reads: a short scan that starts
                // in the middle of a page copies half of it.
                let base = self.row_offsets[self.index] as usize;
                if base > page.len() {
                    return Err(corrupt("bad row offset"));
                }
                self.page_bytes = Bytes::copy_from_slice(&page[base..]);
                self.page_base = base;
                self.loaded_page = Some(self.page);
            }
            if self.index >= self.row_offsets.len() {
                self.page += 1;
                self.index = 0;
                continue;
            }
            let Some(off) = (self.row_offsets[self.index] as usize).checked_sub(self.page_base)
            else {
                return Err(corrupt("cursor moved before its page copy"));
            };
            let raw = parse_row_at(&self.page_bytes, off)?;
            if let Ref::OutOfLine { offset, len } = raw.key {
                self.out_of_line_key = Some(Bytes::copy_from_slice(&file.read(offset, len)?));
            }
            let key: &[u8] = match (&raw.key, &self.out_of_line_key) {
                (Ref::Inline(range), _) => &self.page_bytes[range.clone()],
                (Ref::OutOfLine { .. }, Some(key)) => key,
                (Ref::OutOfLine { .. }, None) => return Err(corrupt("missing out-of-line key")),
            };
            let past_end = match &self.end {
                Bound::Unbounded => false,
                Bound::Included(end) => key > end.as_ref(),
                Bound::Excluded(end) => key >= end.as_ref(),
            };
            if !past_end {
                self.current = Some(raw);
            }
            return Ok(());
        }
    }

    /// The key of the current row, or `None` past the end of the range.
    pub(crate) fn key(&self) -> Option<&[u8]> {
        match self.current.as_ref()?.key {
            Ref::Inline(ref range) => Some(&self.page_bytes[range.clone()]),
            Ref::OutOfLine { .. } => self.out_of_line_key.as_deref(),
        }
    }

    /// Returns the current row as slices of the page buffer, and moves to the next row. The
    /// caller must have seen a key.
    pub(crate) fn take(&mut self, file: &RangeFile) -> io::Result<RowEntry> {
        let Some(raw) = self.current.take() else {
            return Err(corrupt("cursor has no current row"));
        };
        let key = match &raw.key {
            Ref::Inline(range) => self.page_bytes.slice(range.clone()),
            Ref::OutOfLine { .. } => self
                .out_of_line_key
                .take()
                .ok_or_else(|| corrupt("missing out-of-line key"))?,
        };
        let value = match &raw.value {
            Ref::Inline(range) => self.page_bytes.slice(range.clone()),
            Ref::OutOfLine { offset, len } => Bytes::copy_from_slice(&file.read(*offset, *len)?),
        };
        let row = RowEntry {
            key,
            value: if raw.flags & FLAG_MERGE != 0 {
                ValueDeletable::Merge(value)
            } else {
                ValueDeletable::Value(value)
            },
            seq: raw.seq,
            create_ts: (raw.flags & FLAG_CREATE_TS != 0).then_some(raw.create_ts),
            expire_ts: None,
        };
        self.index += 1;
        self.normalize(file)?;
        Ok(row)
    }

    /// Moves to the next row.
    pub(crate) fn skip(&mut self, file: &RangeFile) -> io::Result<()> {
        if self.current.is_some() {
            self.index += 1;
            self.normalize(file)?;
        }
        Ok(())
    }

    /// Moves forward to the first row at or after `key`. It never moves backwards.
    pub(crate) fn seek(&mut self, file: &RangeFile, key: &[u8]) -> io::Result<()> {
        if self.key().is_none_or(|current| current >= key) {
            return Ok(());
        }
        let (page, index) = file.locate(&mut self.reader, Bound::Included(key))?;
        if (page, index) > (self.page, self.index) {
            self.page = page;
            self.index = index;
        }
        self.normalize(file)
    }
}

/// A [`RowEntryIterator`] over a [`RangeFileCursor`]. Rebuilds and the normal iterator stack
/// use it.
pub(crate) struct RangeFileIterator {
    file: Arc<RangeFile>,
    range: BytesRange,
    cursor: Option<RangeFileCursor>,
}

impl RangeFileIterator {
    pub(crate) fn new(file: Arc<RangeFile>, range: BytesRange) -> Self {
        Self {
            file,
            range,
            cursor: None,
        }
    }

    /// The cursor, and the file that it reads.
    fn cursor(&mut self) -> Result<(&mut RangeFileCursor, &RangeFile), SlateDBError> {
        match self.cursor.as_mut() {
            Some(cursor) => Ok((cursor, &self.file)),
            None => Err(SlateDBError::IteratorNotInitialized),
        }
    }
}

#[async_trait]
impl RowEntryIterator for RangeFileIterator {
    async fn init(&mut self) -> Result<(), SlateDBError> {
        if self.cursor.is_none() {
            self.cursor = Some(RangeFileCursor::new(&self.file, &self.range)?);
        }
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<RowEntry>, SlateDBError> {
        let (cursor, file) = self.cursor()?;
        if cursor.key().is_none() {
            return Ok(None);
        }
        Ok(Some(cursor.take(file)?))
    }

    async fn seek(&mut self, next_key: &[u8]) -> Result<(), SlateDBError> {
        let (cursor, file) = self.cursor()?;
        Ok(cursor.seek(file, next_key)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn row(key: &[u8], value: &[u8], seq: u64) -> RowEntry {
        RowEntry::new_value(key, value, seq)
    }

    fn build(
        dir: &Path,
        name: &str,
        page_size: usize,
        source: RangeCachePageSource,
        rows: &[RowEntry],
    ) -> Arc<RangeFile> {
        let mut writer = RangeFileWriter::create(dir, name, page_size).unwrap();
        for r in rows {
            writer.push(r).unwrap();
        }
        let path = writer.finish().unwrap();
        Arc::new(RangeFile::open(path, source, Arc::new(AtomicU64::new(0))).unwrap())
    }

    fn key(i: usize) -> Vec<u8> {
        format!("key{:06}", i * 2).into_bytes()
    }

    fn rows(n: usize, value_len: impl Fn(usize) -> usize) -> Vec<RowEntry> {
        (0..n)
            .map(|i| {
                row(
                    &key(i),
                    &vec![(i % 251) as u8; value_len(i)],
                    1000 + i as u64,
                )
            })
            .collect()
    }

    async fn scan(file: &Arc<RangeFile>, range: BytesRange) -> Vec<RowEntry> {
        let mut iter = RangeFileIterator::new(file.clone(), range);
        iter.init().await.unwrap();
        let mut out = Vec::new();
        while let Some(r) = iter.next().await.unwrap() {
            out.push(r);
        }
        out
    }

    #[rstest]
    #[case::mmap(RangeCachePageSource::Mmap)]
    #[case::positional_read(RangeCachePageSource::PositionalRead)]
    #[tokio::test]
    async fn should_round_trip_rows(#[case] source: RangeCachePageSource) {
        let dir = tempfile::tempdir().unwrap();
        let rows = rows(2000, |i| i % 97);
        let file = build(dir.path(), "rt", 4096, source, &rows);
        assert_eq!(file.rows(), 2000);
        assert!(file.page_count() > 10);
        assert_eq!(file.max_seq(), 2999);
        assert_eq!(scan(&file, BytesRange::unbounded()).await, rows);
        for r in &rows {
            assert_eq!(file.get(&r.key).unwrap().as_ref(), Some(r));
        }
        assert!(!dir.path().join("rt.rcf.tmp").exists());
    }

    #[rstest]
    #[case::mmap(RangeCachePageSource::Mmap)]
    #[case::positional_read(RangeCachePageSource::PositionalRead)]
    #[tokio::test]
    async fn should_find_absent_keys_at_page_boundaries(#[case] source: RangeCachePageSource) {
        let dir = tempfile::tempdir().unwrap();
        let rows = rows(3000, |i| 10 + i % 40);
        let file = build(dir.path(), "pb", 512, source, &rows);
        assert!(file.page_count() > 100);
        for p in 0..file.page_count() {
            let page = file.page(p).unwrap();
            let count = RangeFile::row_count(&page);
            let first = file.key_at(&page, 0).unwrap().to_vec();
            let last = file.key_at(&page, count - 1).unwrap().to_vec();
            assert_eq!(first, file.fence(p));
            assert!(file.get(&first).unwrap().is_some());
            assert!(file.get(&last).unwrap().is_some());
            // Absent keys in the gap before the first key and after the last key of a page.
            let mut before = first.clone();
            *before.last_mut().unwrap() -= 1;
            assert!(file.get(&before).unwrap().is_none());
            let mut after = last.clone();
            after.push(0);
            assert!(file.get(&after).unwrap().is_none());
            let next = scan(
                &file,
                BytesRange::new(Bound::Included(Bytes::from(after)), Bound::Unbounded),
            )
            .await;
            match next.first() {
                Some(r) => assert_eq!(r.key.as_ref(), file.fence(p + 1)),
                None => assert_eq!(p + 1, file.page_count()),
            }
        }
    }

    #[tokio::test]
    async fn should_honor_scan_bounds_and_seek_forward_only() {
        let dir = tempfile::tempdir().unwrap();
        let rows = rows(500, |_| 30);
        let file = build(dir.path(), "seek", 1024, RangeCachePageSource::Mmap, &rows);
        let k = |i: usize| Bytes::from(key(i));
        let got = scan(
            &file,
            BytesRange::new(Bound::Excluded(k(10)), Bound::Included(k(20))),
        )
        .await;
        assert_eq!(got, rows[11..=20]);
        let got = scan(
            &file,
            BytesRange::new(Bound::Included(k(10)), Bound::Excluded(k(20))),
        )
        .await;
        assert_eq!(got, rows[10..20]);

        let mut iter = RangeFileIterator::new(file.clone(), BytesRange::unbounded());
        iter.init().await.unwrap();
        iter.seek(&k(100)).await.unwrap();
        assert_eq!(iter.next().await.unwrap().unwrap().key, k(100));
        // A seek to an earlier key does not move the iterator back.
        iter.seek(&k(50)).await.unwrap();
        assert_eq!(iter.next().await.unwrap().unwrap().key, k(101));
        // A seek to an absent key lands on the next row.
        let mut between = key(300);
        between.push(b'!');
        iter.seek(&between).await.unwrap();
        assert_eq!(iter.next().await.unwrap().unwrap().key, k(301));
        iter.seek(b"zzz").await.unwrap();
        assert!(iter.next().await.unwrap().is_none());

        // At a page boundary, a seek to an earlier key does not move the iterator back.
        let mut iter = RangeFileIterator::new(file.clone(), BytesRange::unbounded());
        iter.init().await.unwrap();
        let first_page_rows = {
            let page = file.page(0).unwrap();
            RangeFile::row_count(&page)
        };
        for _ in 0..first_page_rows {
            iter.next().await.unwrap().unwrap();
        }
        iter.seek(&k(0)).await.unwrap();
        assert_eq!(iter.next().await.unwrap().unwrap().key, k(first_page_rows));
    }

    #[rstest]
    #[case::mmap(RangeCachePageSource::Mmap)]
    #[case::positional_read(RangeCachePageSource::PositionalRead)]
    #[tokio::test]
    async fn should_store_large_keys_and_values_out_of_line(#[case] source: RangeCachePageSource) {
        let dir = tempfile::tempdir().unwrap();
        let page = 512;
        let mut rows = Vec::new();
        for i in 0..200usize {
            let value_len = match i % 6 {
                0 => 0,
                1 => 100,
                2 => page / 4,
                3 => page,
                4 => 3 * page + 7,
                _ => 20_000,
            };
            let mut k = key(i);
            if i % 5 == 0 {
                k.extend(std::iter::repeat_n(b'x', 70 + i));
            }
            let v: Vec<u8> = (0..value_len).map(|j| (j * 31 + i) as u8).collect();
            rows.push(row(&k, &v, i as u64));
        }
        rows.sort_by(|a, b| a.key.cmp(&b.key));
        let file = build(dir.path(), "large", page, source, &rows);
        assert_eq!(scan(&file, BytesRange::unbounded()).await, rows);
        for r in &rows {
            assert_eq!(file.get(&r.key).unwrap().as_ref(), Some(r));
        }
        assert!(file.page_offsets.iter().all(|o| o % page as u64 == 0));
    }

    #[tokio::test]
    async fn should_keep_merge_kind_and_create_ts() {
        let dir = tempfile::tempdir().unwrap();
        let rows = vec![
            row(b"a", b"1", 1).with_create_ts(-5),
            RowEntry::new_merge(b"b", b"2", 2),
            RowEntry::new_merge(b"c", b"3", 3).with_create_ts(7),
            row(b"d", b"", 4),
        ];
        let file = build(dir.path(), "kinds", 4096, RangeCachePageSource::Mmap, &rows);
        assert_eq!(scan(&file, BytesRange::unbounded()).await, rows);
        assert_eq!(file.get(b"c").unwrap(), Some(rows[2].clone()));
    }

    #[tokio::test]
    async fn should_open_an_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = build(dir.path(), "empty", 4096, RangeCachePageSource::Mmap, &[]);
        assert_eq!(file.rows(), 0);
        assert_eq!(file.file_len(), TRAILER_LEN as u64);
        assert!(file.get(b"anything").unwrap().is_none());
        assert!(scan(&file, BytesRange::unbounded()).await.is_empty());
    }

    #[tokio::test]
    async fn should_delete_the_file_when_the_last_reference_drops() {
        let dir = tempfile::tempdir().unwrap();
        let disk_bytes = Arc::new(AtomicU64::new(0));
        let mut writer = RangeFileWriter::create(dir.path(), "life", 4096).unwrap();
        writer.push(&row(b"a", b"1", 1)).unwrap();
        assert!(dir.path().join("life.rcf.tmp").exists());
        let path = writer.finish().unwrap();
        let file = Arc::new(
            RangeFile::open(path.clone(), RangeCachePageSource::Mmap, disk_bytes.clone()).unwrap(),
        );
        assert_eq!(disk_bytes.load(Ordering::Relaxed), file.file_len());
        let mut iter = RangeFileIterator::new(file.clone(), BytesRange::unbounded());
        iter.init().await.unwrap();
        drop(file);
        assert!(path.exists(), "an iterator keeps the file");
        assert_eq!(iter.next().await.unwrap().unwrap().key.as_ref(), b"a");
        drop(iter);
        assert!(!path.exists(), "the last reference deletes the file");
        assert_eq!(disk_bytes.load(Ordering::Relaxed), 0);

        // A writer that does not finish removes its temporary file.
        let mut writer = RangeFileWriter::create(dir.path(), "abandoned", 4096).unwrap();
        writer.push(&row(b"a", b"1", 1)).unwrap();
        drop(writer);
        assert!(!dir.path().join("abandoned.rcf.tmp").exists());
    }

    #[test]
    fn should_count_pages_by_fence_with_prefixes() {
        // The keys share "tenant/00", and blocks of 50 share eight more bytes, so the prefix
        // search often ties and compares whole keys.
        let dir = tempfile::tempdir().unwrap();
        let keys: Vec<Vec<u8>> = (0..600)
            .map(|i| format!("tenant/{:04}/{:08}", i / 50, i).into_bytes())
            .collect();
        let rows: Vec<RowEntry> = keys.iter().map(|k| row(k, b"v", 1)).collect();
        let file = build(dir.path(), "fp", 512, RangeCachePageSource::Mmap, &rows);
        assert!(file.page_count() > 20);
        let fences: Vec<Vec<u8>> = (0..file.page_count())
            .map(|p| file.fence(p).to_vec())
            .collect();
        let mut probes: Vec<Vec<u8>> = Vec::new();
        for k in &keys {
            probes.push(k.clone());
            let mut longer = k.clone();
            longer.push(0);
            probes.push(longer);
            probes.push(k[..k.len() - 1].to_vec());
        }
        for p in [
            &b""[..],
            b"a",
            b"t",
            b"tenant",
            b"tenant/0",
            b"tenant/00",
            b"tenant/01",
            b"tenant/0011/99999999",
            b"tenant/1",
            b"zzz",
        ] {
            probes.push(p.to_vec());
        }
        for probe in &probes {
            let expected = fences
                .iter()
                .filter(|f| f.as_slice() <= probe.as_slice())
                .count();
            assert_eq!(
                file.pages_le(probe),
                expected,
                "{:?}",
                String::from_utf8_lossy(probe)
            );
            // The in-page search, through point reads.
            let present = keys.contains(probe);
            assert_eq!(file.get(probe).unwrap().is_some(), present);
        }
    }

    #[rstest]
    #[case::short_prefix(0)]
    #[case::prefix_longer_than_hints_skip(70)]
    #[tokio::test]
    async fn should_search_pages_by_key_hints(#[case] prefix_len: usize) {
        // Keys of different lengths, some equal in their first bytes after a shared prefix, so
        // hints tie and the search compares whole keys.
        let dir = tempfile::tempdir().unwrap();
        let prefix = "p".repeat(prefix_len);
        let mut keys: Vec<Vec<u8>> = (0..400)
            .flat_map(|i| {
                let base = format!("{prefix}{:03}", i / 4);
                [
                    base.clone(),
                    format!("{base}{:02}", i % 4),
                    format!("{base}{:02}zz", i % 4),
                ]
            })
            .map(String::into_bytes)
            .collect();
        keys.sort();
        keys.dedup();
        let rows: Vec<RowEntry> = keys.iter().map(|k| row(k, b"value", 1)).collect();
        let file = build(dir.path(), "kh", 512, RangeCachePageSource::Mmap, &rows);
        for (i, k) in keys.iter().enumerate() {
            assert_eq!(
                file.get(k).unwrap().map(|r| r.key),
                Some(Bytes::copy_from_slice(k))
            );
            let mut after = k.clone();
            after.push(0);
            assert!(file.get(&after).unwrap().is_none());
            // A scan from just after this key starts at the next key.
            let next = scan(
                &file,
                BytesRange::new(Bound::Excluded(Bytes::copy_from_slice(k)), Bound::Unbounded),
            )
            .await;
            assert_eq!(
                next.first().map(|r| r.key.to_vec()),
                keys.get(i + 1).cloned()
            );
        }
    }

    #[test]
    fn should_reject_bad_rows_and_page_sizes() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = RangeFileWriter::create(dir.path(), "order", 4096).unwrap();
        writer.push(&row(b"b", b"", 1)).unwrap();
        assert!(writer.push(&row(b"b", b"", 2)).is_err());
        assert!(writer.push(&row(b"a", b"", 2)).is_err());
        assert!(writer.push(&RowEntry::new_tombstone(b"c", 2)).is_err());
        assert!(writer.push(&row(b"d", b"", 2).with_expire_ts(10)).is_err());
        assert!(RangeFileWriter::create(dir.path(), "p", 1000).is_err());
        assert!(RangeFileWriter::create(dir.path(), "p", 256).is_err());
        assert!(RangeFileWriter::create(dir.path(), "p", 131072).is_err());
    }

    #[test]
    fn should_reject_a_corrupt_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.rcf");
        fs::write(&path, vec![7u8; 200]).unwrap();
        let err = RangeFile::open(
            path.clone(),
            RangeCachePageSource::PositionalRead,
            Arc::new(AtomicU64::new(0)),
        )
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(!path.exists(), "a file that fails to open is removed");
    }
}
