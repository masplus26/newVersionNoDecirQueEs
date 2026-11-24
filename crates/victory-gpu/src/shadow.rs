//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::vram::Vram;
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    /// What shaders use: `data`'s own buffer, or one in video memory.
    pub(crate) buffer: ID3D12Resource,
    vram: Option<Vram>,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// The ranges shaders of a submission not yet copied back may be
    /// writing, with the copy's bytes there before it ran (what the GPU
    /// wrote differs): disjoint segments by offset. Title memory there is
    /// older than the copy.
    before: Mutex<BTreeMap<usize, Segment>>,
}

/// A range shaders may write: its length, and its bytes before the
/// submission (empty in video memory, where the GPU keeps them).
struct Segment {
    len: usize,
    bytes: Vec<u8>,
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// Smaller snapshots are plain allocations: pooling them would push the big
/// ones out.
const POOLED: usize = 256 << 10;

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    if n < POOLED {
        return Vec::with_capacity(n);
    }
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest).
fn give_back(v: Vec<u8>) {
    if v.capacity() < POOLED {
        return;
    }
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > 8 {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(8);
    }
}

/// The parts of `[from, to)` outside `segments`, in order.
fn uncovered(segments: &BTreeMap<usize, Segment>, from: usize, to: usize) -> Vec<(usize, usize)> {
    let mut gaps = Vec::new();
    // Only the last segment starting at or before `from` can reach into it.
    let mut at = segments
        .range(..=from)
        .next_back()
        .map_or(from, |(&start, s)| from.max(start + s.len));
    if at < to {
        for (&start, segment) in segments.range(at..to) {
            if start > at {
                gaps.push((at, start));
            }
            at = start + segment.len;
        }
    }
    if at < to {
        gaps.push((at, to));
    }
    gaps
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    /// The video memory side, if it has one.
    pub(crate) fn vram(&self) -> Option<&Vram> {
        self.vram.as_ref()
    }

    /// The title address it starts at.
    pub(crate) fn start(&self) -> usize {
        self.start as usize
    }

    /// The ranges of the submission's shaders, as offsets.
    pub(crate) fn segments(&self) -> Vec<(usize, usize)> {
        let before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        before.iter().map(|(&at, s)| (at, at + s.len)).collect()
    }

    /// With `data` the mapping of `staging`, which shaders use unless
    /// `vram` is given.
    fn new(
        staging: ID3D12Resource,
        data: *mut u8,
        vram: Option<(ID3D12Resource, Vram)>,
        start: u64,
        len: u64,
    ) -> Shadow {
        let (buffer, vram) = match vram {
            Some((buffer, vram)) => (buffer, Some(vram)),
            None => (staging, None),
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            vram,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            before: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU
    /// wrote: those not yet kept for this submission only (the world declares
    /// overlapping views of its 174 MB heap dozens of times), in reused
    /// memory (fresh snapshots cost ~2.3 million page faults per world
    /// frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        for (a, b) in uncovered(&before, from, to) {
            let mut bytes = Vec::new();
            if self.vram.is_none() {
                bytes = pooled(b - a);
                // SAFETY: inside the mapping, which nothing writes now.
                bytes.extend_from_slice(unsafe {
                    std::slice::from_raw_parts(self.data.add(a), b - a)
                });
            }
            before.insert(a, Segment { len: b - a, bytes });
        }
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Every range the
    /// submission declared, each byte once (one submission runs at a time):
    /// a second pass would compare the whole range again, and write bytes
    /// the first took from the title over newer ones. Returns the bytes
    /// written; nothing is left to copy back. What shaders wrote in video
    /// memory was brought back before ([`crate::vram::bring_back`]).
    pub(crate) fn copy_back(&self) -> usize {
        let mut segments = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        for (from, segment) in std::mem::take(&mut *segments) {
            if self.vram.is_none() {
                written += self.copy_back_segment(from, &segment.bytes);
                give_back(segment.bytes);
            }
        }
        written
    }

    /// [`Shadow::copy_back`] of one segment: the copy's bytes `before`, from
    /// offset `from` on.
    fn copy_back_segment(&self, from: usize, before: &[u8]) -> usize {
        // By 64 KiB piece: the copy compared with its bytes before the
        // submission, title memory read only where the GPU wrote.
        let to = from + before.len();
        let mut title = Vec::new();
        let mut synced = Vec::new();
        let mut written = 0;
        for (at, n) in crate::vram::pieces(self.start as usize, from, to) {
            let old = &before[at - from..at - from + n];
            // SAFETY: inside the mapping; only the submitting thread
            // writes it.
            let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
            if host != old {
                written += self.merge(at, host, old, &mut title, &mut synced);
            }
        }
        self.mark_current(from, to, synced);
        written
    }

    /// Puts the GPU's bytes (those of `host`, the copy's piece at offset
    /// `at`, that differ from `old`) into title memory; the copy takes the
    /// title's bytes in turn. Returns the bytes written, and adds the page
    /// to `synced`.
    fn merge(
        &self,
        at: usize,
        host: &mut [u8],
        old: &[u8],
        title: &mut Vec<u8>,
        synced: &mut Vec<usize>,
    ) -> usize {
        let (address, n) = (self.start as usize + at, host.len());
        title.resize(n, 0);
        if !crate::host::read_into(address as *const u8, title) {
            return 0;
        }
        // Without branches, so that it runs many bytes at a time.
        let mut changed = 0;
        for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
            let merged = if *h != *o { *h } else { *t };
            changed |= merged ^ *t;
            (*t, *h) = (merged, merged);
        }
        synced.push(address / PAGE);
        if changed == 0 {
            return 0;
        }
        crate::watch::writing(address, n);
        crate::host::write_mapped(address as *mut u8, title.as_ptr(), n);
        n
    }

    /// A piece the GPU found changed in video memory: `now`, from `before`
    /// the submission. As [`Shadow::copy_back`]; where the title's bytes
    /// differ from the GPU's, they are uploaded in turn.
    pub(crate) fn take_piece(
        self: &Arc<Self>,
        at: usize,
        now: &[u8],
        before: &[u8],
        scratch: &mut Vec<u8>,
        synced: &mut Vec<usize>,
    ) -> usize {
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), now.len()) };
        host.copy_from_slice(now);
        let written = self.merge(at, host, before, scratch, synced);
        if let Some(vram) = &self.vram
            && host != now
        {
            crate::vram::changed(self, vram, at, at + now.len());
        }
        written
    }

    /// Pages made equal to title memory are current in the copy, so our own
    /// write is not taken for the title's; others keep what they had (a
    /// title write there shows at the next refresh). Only pages wholly in
    /// `[from, to)`: the rest of a page was not made equal.
    pub(crate) fn mark_current(&self, from: usize, to: usize, synced: Vec<usize>) {
        let start = self.start as usize;
        let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
        if first < last
            && !synced.is_empty()
            && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
            && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
        {
            let base = start / PAGE;
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                copied[page - base] = now;
            }
        }
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(self: &Arc<Self>, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        // Around ranges shaders are writing: the gaps, each as below (in
        // the world most submissions have such ranges; copying the gaps
        // whole cost ~1 s per frame).
        let gaps = {
            let before = self.before.lock().unwrap_or_else(|e| e.into_inner());
            uncovered(&before, from - start, to - start)
        };
        gaps.into_iter()
            .map(|(a, b)| self.update_free(start + a, start + b))
            .sum()
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(self: &Arc<Self>, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(self: &Arc<Self>, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(self: &Arc<Self>, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        if let Some(vram) = &self.vram {
            crate::vram::changed(self, vram, from - start, to - start);
        }
        to - from
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let host = host()?;
        let room = len.next_multiple_of(1 << 16);
        let (staging, data) = host.shared_buffer(room)?;
        let vram = crate::vram::make(host, &staging, key.0, room);
        crate::host::name(vram.as_ref().map_or(&staging, |(b, _)| b), || {
            format!("title buffer {:#x}+{len:#x}", key.0)
        });
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(staging, data, vram, key.0, len)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, None, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncovered_parts_of_ranges() {
        let segment = |len| Segment { len, bytes: Vec::new() };
        let map = BTreeMap::from([(10, segment(10)), (20, segment(5)), (40, segment(10))]);
        assert_eq!(uncovered(&BTreeMap::new(), 3, 9), [(3, 9)]);
        assert_eq!(uncovered(&map, 0, 60), [(0, 10), (25, 40), (50, 60)]);
        assert_eq!(uncovered(&map, 15, 45), [(25, 40)]);
        assert_eq!(uncovered(&map, 12, 24), []);
        assert_eq!(uncovered(&map, 25, 40), [(25, 40)]);
        assert_eq!(uncovered(&map, 30, 30), []);
        assert_eq!(uncovered(&map, 15, 15), []);
    }

    const PAGES: usize = 4;

    /// Write-watched title memory and its copy; none without a GPU.
    fn watched(bytes: usize) -> Option<(&'static mut [u8], Arc<Shadow>)> {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let (buffer, data) = host()?.shared_buffer(bytes as u64)?;
        // SAFETY: a fresh allocation, kept to the end of the test process,
        // and a live buffer.
        unsafe {
            let flags = MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH;
            let p = VirtualAlloc(std::ptr::null(), bytes, flags, PAGE_READWRITE).cast::<u8>();
            assert!(!p.is_null());
            let shadow = Shadow::new(buffer, data, None, p as u64, bytes as u64);
            Some((std::slice::from_raw_parts_mut(p, bytes), Arc::new(shadow)))
        }
    }

    /// Overlapping ranges: the GPU's bytes reach the title, the title's own
    /// writes meanwhile survive, and the copy ends up current.
    #[test]
    fn shader_writes_and_title_writes_both_survive() {
        let Some((title, shadow)) = watched(PAGES * PAGE) else { return };
        let at = title.as_ptr() as usize;
        title.fill(1);
        assert_eq!(shadow.update(at, PAGES * PAGE), PAGES * PAGE);
        shadow.writing(at + PAGE, PAGE + 100);
        shadow.writing(at + PAGE / 2, 2 * PAGE);
        shadow.writing(at + PAGE, PAGE + 100);
        let kept: usize = shadow.before.lock().unwrap().values().map(|s| s.bytes.len()).sum();
        assert_eq!(kept, 2 * PAGE, "each byte kept once");
        // SAFETY: inside the mapping; stands in for the GPU.
        let gpu = |i: usize, v: u8| unsafe { shadow.data.add(i).write(v) };
        for i in [PAGE / 2, PAGE, 2 * PAGE + 99, 2 * PAGE + 100] {
            gpu(i, 9);
        }
        title[PAGE + 5] = 7;
        title[3 * PAGE] = 8;
        // A refresh meanwhile leaves what shaders write alone.
        shadow.update(at, PAGES * PAGE);
        shadow.copy_back();
        for i in [PAGE / 2, PAGE, 2 * PAGE + 99, 2 * PAGE + 100] {
            assert_eq!(title[i], 9, "{i}");
        }
        assert_eq!((title[PAGE + 5], title[3 * PAGE], title[PAGE + 6]), (7, 8, 1));
        assert!(shadow.before.lock().unwrap().is_empty());
        shadow.update(at, PAGES * PAGE);
        // SAFETY: the mapping.
        assert!(unsafe { std::slice::from_raw_parts(shadow.data, PAGES * PAGE) } == title);
    }

    /// `cargo test --release -p victory-gpu overlapping_views -- --ignored --nocapture`
    #[test]
    #[ignore = "a benchmark"]
    fn overlapping_views_of_a_big_heap() {
        const LEN: usize = 174 << 20;
        const VIEWS: usize = 24;
        const ROUNDS: u32 = 10;
        let Some((title, shadow)) = watched(LEN) else { return };
        let at = title.as_ptr() as usize;
        title.fill(1);
        shadow.update(at, LEN);
        let started = std::time::Instant::now();
        for _ in 0..ROUNDS {
            for v in 0..VIEWS {
                shadow.writing(at + v * PAGE, LEN - 2 * v * PAGE);
            }
            shadow.copy_back();
        }
        eprintln!("{VIEWS} overlapping views: {:?} per submission", started.elapsed() / ROUNDS);
        let started = std::time::Instant::now();
        for _ in 0..ROUNDS {
            shadow.writing(at, PAGE);
            shadow.writing(at + LEN - PAGE, PAGE);
            shadow.copy_back();
        }
        eprintln!("two distant views: {:?} per submission", started.elapsed() / ROUNDS);
    }
}
