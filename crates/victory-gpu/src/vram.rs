//! Host copies of big title buffers in video memory ([`crate::shadow`]).
//! Shaders read and write them at video memory's speed, and the GPU finds
//! what a submission's shaders wrote: the ranges they may write are copied
//! aside (video memory to video memory) before the submission, compared
//! after it, and only the 64 KiB pieces that differ travel to the CPU.
//!
//! The CPU's side of such a copy is system memory of the same size: the
//! title's data lands there, and what changed of it is uploaded before the
//! next submission. Copies made in system memory (small buffers, GPUs
//! without memory of their own, no room) are compared by the CPU.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
};
use windows::core::{Interface, s};

use crate::host::{Host, HostList, transition};
use crate::shadow::Shadow;
use crate::watch::PAGE;

/// The unit compared and brought back.
pub(crate) const PIECE: usize = PAGE;
/// Smaller buffers stay in system memory: the CPU compares them faster than
/// a round trip takes.
const MIN_BYTES: u64 = 4 << 20;
/// The comparison addresses buffers with 32 bits.
const MAX_BYTES: u64 = 1 << 30;
/// Of the GPU's own memory, the part these copies may take (one in this).
const SHARE: u64 = 4;
/// Pieces brought back per round trip.
const ROUND: usize = 256;

const SOURCE: &str = r#"
#define ROOT "RootConstants(num32BitConstants=4, b0), SRV(t0), SRV(t1), UAV(u0)"
cbuffer C : register(b0) { uint first, last, phase, slot; };
ByteAddressBuffer now : register(t0);
ByteAddressBuffer before : register(t1);
RWByteAddressBuffer mask : register(u0);
groupshared uint differs;

// One group per 64 KiB piece of title memory (`phase`: where the buffer
// starts in its piece), cut to [first, last).
[RootSignature(ROOT)]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const uint PIECE = 65536;
    if (thread == 0) differs = 0;
    GroupMemoryBarrierWithGroupSync();
    uint start = ((first + phase) / PIECE + group.x) * PIECE;
    uint from = max(start, first + phase) - phase;
    uint to = min(start + PIECE, last + phase) - phase;
    bool found = false;
    for (uint at = from + thread * 16; at < to; at += 64 * 16) {
        if (any(now.Load4(at) != before.Load4(at))) found = true;
    }
    if (found) InterlockedOr(differs, 1);
    GroupMemoryBarrierWithGroupSync();
    if (thread == 0) mask.Store((slot + group.x) * 4, differs);
}
"#;

/// The video memory side of a [`Shadow`].
pub(crate) struct Vram {
    /// The CPU's side (the shadow's mapped memory).
    staging: ID3D12Resource,
    bytes: u64,
    /// What the ranges shaders may write held before the submission. Made
    /// with the copy: without it, what shaders write could not be told.
    before: ID3D12Resource,
    /// Offsets of the CPU's side changed since the last upload.
    dirty: Mutex<Vec<(usize, usize)>>,
}

// SAFETY: D3D12 resources are free-threaded.
unsafe impl Send for Vram {}
// SAFETY: as above.
unsafe impl Sync for Vram {}

impl Drop for Vram {
    fn drop(&mut self) {
        USED.fetch_sub(2 * (self.bytes + PIECE as u64), Ordering::Relaxed);
    }
}

/// Video memory these copies take.
static USED: AtomicU64 = AtomicU64::new(0);

/// How much they may take: none on GPUs that share the system's memory.
fn budget(host: &Host) -> u64 {
    static BUDGET: OnceLock<u64> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let mut memory = D3D12_FEATURE_DATA_ARCHITECTURE1::default();
        // SAFETY: plain queries; the output is of the given size.
        let dedicated = unsafe {
            let asked = host.device.CheckFeatureSupport(
                D3D12_FEATURE_ARCHITECTURE1,
                (&raw mut memory).cast(),
                size_of_val(&memory) as u32,
            );
            if asked.is_err() || memory.UMA.as_bool() {
                return 0;
            }
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid()))
                .and_then(|a| a.GetDesc1())
                .map_or(0, |d| d.DedicatedVideoMemory as u64)
        };
        tracing::info!(bytes = dedicated / SHARE, "video memory for title buffers");
        dedicated / SHARE
    })
}

/// A buffer of `bytes` in video memory, within the budget.
fn allocate(host: &Host, bytes: u64) -> Option<ID3D12Resource> {
    // Room for the comparison's last load to run past the end.
    let bytes = bytes + PIECE as u64;
    if USED.fetch_add(bytes, Ordering::Relaxed) + bytes > budget(host) {
        USED.fetch_sub(bytes, Ordering::Relaxed);
        return None;
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut made: Option<ID3D12Resource> = None;
    // SAFETY: plain creation.
    let result = unsafe {
        host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut made,
        )
    };
    if let Err(e) = result {
        tracing::warn!(error = %e, bytes, "no video memory for a title buffer");
    }
    if made.is_none() {
        USED.fetch_sub(bytes, Ordering::Relaxed);
    }
    made
}

/// The buffer shaders use for a title buffer at `start` whose CPU side is
/// `staging` (`bytes` long), with its state; `None` to use `staging` itself.
pub(crate) fn make(
    host: &Host,
    staging: &ID3D12Resource,
    start: u64,
    bytes: u64,
) -> Option<(ID3D12Resource, Vram)> {
    let fits = (MIN_BYTES..=MAX_BYTES).contains(&bytes) && start.is_multiple_of(4);
    if !fits || tools(host).is_none() {
        return None;
    }
    let buffer = allocate(host, bytes)?;
    let Some(before) = allocate(host, bytes) else {
        USED.fetch_sub(bytes + PIECE as u64, Ordering::Relaxed);
        return None;
    };
    let vram = Vram {
        staging: staging.clone(),
        bytes,
        before,
        dirty: Mutex::default(),
    };
    Some((buffer, vram))
}

/// Shadows whose CPU side changed since the last submission.
static DIRTY: Mutex<Vec<Arc<Shadow>>> = Mutex::new(Vec::new());

fn locked<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The CPU's side changed in `[from, to)` (offsets): uploaded before the
/// next submission runs. Called by the submitting thread.
pub(crate) fn changed(shadow: &Arc<Shadow>, vram: &Vram, from: usize, to: usize) {
    if from >= to {
        return;
    }
    let first = {
        let mut dirty = locked(&vram.dirty);
        dirty.push((from, to));
        dirty.len() == 1
    };
    if first {
        locked(&DIRTY).push(shadow.clone());
    }
}

/// `ranges` in order, overlapping and touching ones joined.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some((_, end)) if from <= *end => *end = (*end).max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// `[from, to)` of a buffer starting at title address `start`, cut at the
/// title's pieces: (offset, bytes).
pub(crate) fn pieces(start: usize, from: usize, to: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity((to - from) / PIECE + 2);
    let mut at = from;
    while at < to {
        let n = (PIECE - (start + at) % PIECE).min(to - at);
        out.push((at, n));
        at += n;
    }
    out
}

/// What the submissions share; used by one at a time.
struct Tools {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    work: Mutex<Work>,
}

struct Work {
    upload: HostList,
    compare: HostList,
    fetch: HostList,
    /// A dword per compared piece, and its capacity.
    mask: Option<(ID3D12Resource, *mut u8, usize)>,
    /// `ROUND` pieces as they are, then as they were.
    fetched: (ID3D12Resource, *mut u8),
}

// SAFETY: D3D12 objects are free-threaded; the mappings are used under
// `work`'s lock.
unsafe impl Send for Tools {}
// SAFETY: as above.
unsafe impl Sync for Tools {}

fn tools(host: &Host) -> Option<&'static Tools> {
    static TOOLS: OnceLock<Option<Tools>> = OnceLock::new();
    TOOLS
        .get_or_init(|| match make_tools(host) {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::warn!(error = %e, "no GPU comparison: title buffers stay in system memory");
                None
            }
        })
        .as_ref()
}

fn make_tools(host: &Host) -> windows::core::Result<Tools> {
    let none = windows::core::Error::empty;
    let (root, pso) = make_comparison(host)?;
    let list = || {
        let list = host.new_list()?;
        list.close();
        windows::core::Result::Ok(list)
    };
    Ok(Tools {
        root,
        pso,
        work: Mutex::new(Work {
            upload: list()?,
            compare: list()?,
            fetch: list()?,
            mask: None,
            fetched: host
                .shared_buffer(2 * (ROUND * PIECE) as u64)
                .ok_or_else(none)?,
        }),
    })
}

fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: the blob's own bytes, borrowed with it.
    unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) }
}

fn make_comparison(
    host: &Host,
) -> windows::core::Result<(ID3D12RootSignature, ID3D12PipelineState)> {
    let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("compare"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        let text = errors.map(|b| String::from_utf8_lossy(blob_bytes(&b)).into_owned());
        tracing::warn!(?text, "the comparison does not compile");
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    let bytes = blob_bytes(&code);
    // SAFETY: plain creation; the root signature is inside the shader.
    unsafe {
        let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes)?;
        let pso = host
            .device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
        Ok((root, pso))
    }
}

/// A segment compared: its shadow, its offsets and its first mask dword.
struct Compared {
    shadow: Arc<Shadow>,
    from: usize,
    to: usize,
    slot: usize,
}

/// The lists to run around a submission's own, and what they compare.
#[derive(Default)]
pub(crate) struct Around {
    pub(crate) first: Option<ID3D12CommandList>,
    pub(crate) last: Option<ID3D12CommandList>,
    compared: Vec<Compared>,
}

/// Before a submission whose shaders write `written`: records the uploads
/// and the copies aside, and the comparison after it.
pub(crate) fn around(host: &Host, written: &[Arc<Shadow>]) -> Around {
    let dirty = std::mem::take(&mut *locked(&DIRTY));
    let compared = to_compare(written);
    let mut around = Around::default();
    if dirty.is_empty() && compared.is_empty() {
        return around;
    }
    let Some(tools) = tools(host) else {
        return around;
    };
    let mut work = locked(&tools.work);
    let list = &work.upload;
    list.reset();
    for shadow in &dirty {
        if let Some(vram) = shadow.vram() {
            upload(list, shadow, vram);
        }
    }
    for c in &compared {
        if let Some(vram) = c.shadow.vram() {
            let (from, len) = (c.from as u64, (c.to - c.from) as u64);
            // SAFETY: an open list; ranges inside live buffers, which the
            // written lists' steps hold.
            unsafe { list.list.CopyBufferRegion(&vram.before, from, &c.shadow.buffer, from, len) };
        }
    }
    list.close();
    around.first = list.list.cast().ok();
    let slots = compared
        .last()
        .map_or(0, |c| c.slot + pieces(c.shadow.start(), c.from, c.to).len());
    around.compared = compared;
    if slots > 0 {
        if !work.room_for(host, slots) {
            // Going on would lose what the shaders write.
            crate::host::fail(format_args!("no memory to compare {slots} pieces"));
        }
        around.last = record_comparison(tools, &work, &around.compared);
    }
    around
}

/// The segments of `written` in video memory, with their mask slots.
fn to_compare(written: &[Arc<Shadow>]) -> Vec<Compared> {
    let mut compared = Vec::new();
    let mut slot = 0;
    for shadow in written {
        if shadow.vram().is_none() {
            continue;
        }
        for (from, to) in shadow.segments() {
            compared.push(Compared {
                shadow: shadow.clone(),
                from,
                to,
                slot,
            });
            slot += pieces(shadow.start(), from, to).len();
        }
    }
    compared
}

/// Records the upload of what changed on the CPU's side, leaving the buffer
/// ready to be copied from.
fn upload(list: &HostList, shadow: &Shadow, vram: &Vram) {
    for (from, to) in joined(std::mem::take(&mut *locked(&vram.dirty))) {
        let (from, len) = (from as u64, (to - from) as u64);
        // SAFETY: an open list; ranges inside live buffers.
        unsafe {
            list.list
                .CopyBufferRegion(&shadow.buffer, from, &vram.staging, from, len);
        }
    }
    let readable = transition(
        &shadow.buffer,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    // SAFETY: an open list; the copies above left the buffer a destination.
    unsafe { list.list.ResourceBarrier(&[readable]) };
    // Alive until the list is reset, that is, past its execution.
    list.keep(&shadow.buffer);
    list.keep(&vram.staging);
}

fn record_comparison(
    tools: &Tools,
    work: &Work,
    compared: &[Compared],
) -> Option<ID3D12CommandList> {
    let (mask, ..) = work.mask.as_ref()?;
    let list = &work.compare;
    list.reset();
    // SAFETY: an open list and live objects, kept past its execution.
    unsafe {
        let l = &list.list;
        l.SetComputeRootSignature(&tools.root);
        l.SetPipelineState(&tools.pso);
        l.SetComputeRootUnorderedAccessView(3, mask.GetGPUVirtualAddress());
        for c in compared {
            let Some(vram) = c.shadow.vram() else {
                continue;
            };
            // Whole dwords: buffers start on one (`make`).
            let (first, last) = (c.from / 4 * 4, c.to.next_multiple_of(4));
            let phase = c.shadow.start() % PIECE;
            let constants = [first, last, phase, c.slot].map(|n| n as u32);
            l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, c.shadow.buffer.GetGPUVirtualAddress());
            l.SetComputeRootShaderResourceView(2, vram.before.GetGPUVirtualAddress());
            let groups = pieces(c.shadow.start(), c.from, c.to).len();
            l.Dispatch(groups as u32, 1, 1);
        }
    }
    list.close();
    list.list.cast().ok()
}

impl Work {
    /// Makes the mask hold `slots` dwords.
    fn room_for(&mut self, host: &Host, slots: usize) -> bool {
        if self.mask.as_ref().is_none_or(|(.., n)| *n < slots) {
            let slots = slots.next_power_of_two().max(4096);
            self.mask = host
                .shared_buffer(4 * slots as u64)
                .map(|(buffer, data)| (buffer, data, slots));
        }
        self.mask.is_some()
    }
}

/// After the submission ran: brings the pieces that differ and puts them
/// into title memory (`run` executes a list and waits for it). Returns the
/// bytes written there.
pub(crate) fn bring_back(
    host: &Host,
    around: &Around,
    mut run: impl FnMut(ID3D12CommandList),
) -> usize {
    if around.last.is_none() {
        return 0;
    }
    let Some(tools) = tools(host) else { return 0 };
    let work = locked(&tools.work);
    let Some((_, mask, _)) = &work.mask else {
        return 0;
    };
    let (mut written, mut scratch) = (0, Vec::with_capacity(PIECE));
    for c in &around.compared {
        let differing: Vec<(usize, usize)> = pieces(c.shadow.start(), c.from, c.to)
            .into_iter()
            .enumerate()
            // SAFETY: the mask holds a dword per piece, written by GPU work
            // that was waited for.
            .filter(|(i, _)| unsafe { mask.cast::<u32>().add(c.slot + i).read_volatile() } != 0)
            .map(|(_, piece)| piece)
            .collect();
        let mut synced = Vec::new();
        for round in differing.chunks(ROUND) {
            let Some(list) = record_fetch(&work, c, round) else {
                continue;
            };
            run(list);
            for (i, &(at, n)) in round.iter().enumerate() {
                // SAFETY: the parts of the mapping the list just filled.
                let piece = |slot: usize| unsafe {
                    std::slice::from_raw_parts(work.fetched.1.add(slot * PIECE), n)
                };
                let (now, before) = (piece(i), piece(ROUND + i));
                written += c
                    .shadow
                    .take_piece(at, now, before, &mut scratch, &mut synced);
            }
        }
        c.shadow.mark_current(c.from, c.to, synced);
    }
    written
}

/// Records the copy of `round`'s pieces, as they are and as they were.
fn record_fetch(
    work: &Work,
    c: &Compared,
    round: &[(usize, usize)],
) -> Option<ID3D12CommandList> {
    let before = &c.shadow.vram()?.before;
    let list = &work.fetch;
    list.reset();
    for (i, &(at, n)) in round.iter().enumerate() {
        for (from, slot) in [(&c.shadow.buffer, i), (before, ROUND + i)] {
            let to = (slot * PIECE) as u64;
            // SAFETY: an open list; ranges inside live buffers.
            unsafe {
                list.list
                    .CopyBufferRegion(&work.fetched.0, to, from, at as u64, n as u64);
            }
        }
    }
    list.close();
    list.list.cast().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_are_joined() {
        assert_eq!(joined(vec![]), []);
        assert_eq!(
            joined(vec![(8, 9), (0, 4), (4, 6), (1, 2), (8, 12)]),
            [(0, 6), (8, 12)]
        );
    }

    #[test]
    fn pieces_follow_title_memory() {
        let half = PIECE / 2;
        assert_eq!(pieces(half, 0, PIECE), [(0, half), (half, half)]);
        assert_eq!(pieces(0, 8, PIECE + 4), [(8, PIECE - 8), (PIECE, 4)]);
        assert_eq!(pieces(0, 5, 5), []);
    }

    #[test]
    fn the_comparison_builds() {
        if let Some(host) = crate::host::host() {
            assert!(tools(host).is_some());
        }
    }

    const LEN: usize = 8 << 20;
    /// Not at a piece's start.
    const SKEW: usize = 4100;

    /// A title buffer in write-watched memory and its copy; none without
    /// video memory.
    fn title_buffer() -> Option<(&'static mut [u8], Arc<Shadow>)> {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let flags = MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH;
        // SAFETY: a fresh allocation, kept to the end of the test process.
        let title = unsafe {
            let p = VirtualAlloc(std::ptr::null(), LEN + PIECE, flags, PAGE_READWRITE).cast::<u8>();
            assert!(!p.is_null());
            std::slice::from_raw_parts_mut(p.add(SKEW), LEN)
        };
        title.fill(1);
        let start = title.as_ptr() as u64;
        crate::shadow::register(start as usize, start, LEN as u64);
        let shadow = crate::shadow::find(start, LEN as u64)?;
        shadow.vram().is_some().then_some((title, shadow))
    }

    /// Runs a list in which the GPU stores 9 at the given dwords.
    fn gpu_stores(host: &Host, shadow: &Arc<Shadow>, at: &[usize], meanwhile: impl Fn() + Send + 'static) {
        let list = host.new_list().unwrap();
        let start = shadow.start() as u64;
        shadow.refresh(&list, start, LEN as u64);
        shadow.gpu_writes(&list, start, LEN as u64);
        list.defer(meanwhile);
        let (source, data) = list.upload(host, 4).unwrap();
        // SAFETY: four mapped bytes; an open list and live buffers.
        unsafe {
            data.write_bytes(9, 4);
            for &at in at {
                list.list.CopyBufferRegion(&shadow.buffer, at as u64, &source, 0, 4);
            }
        }
        list.close();
        host.execute(&[&list]);
    }

    /// What shaders see in the buffer.
    fn seen(host: &Host, shadow: &Arc<Shadow>) -> Vec<u8> {
        let list = host.new_list().unwrap();
        let (back, data) = host.shared_buffer(LEN as u64).unwrap();
        // SAFETY: an open list and live buffers; the mapping once it ran.
        unsafe {
            list.list.CopyBufferRegion(&back, 0, &shadow.buffer, 0, LEN as u64);
            list.close();
            host.execute(&[&list]);
            std::slice::from_raw_parts(data, LEN).to_vec()
        }
    }

    /// The first and last dwords of the buffer and of every piece, each in a
    /// submission of its own piece's, reach the title and nothing else does.
    #[test]
    fn stores_anywhere_in_a_piece_are_found() {
        let Some(host) = crate::host::host() else { return };
        let Some((title, shadow)) = title_buffer() else { return };
        let all = pieces(shadow.start(), 0, LEN);
        assert_eq!(all[0].1, PIECE - SKEW);
        let mut expected = title.to_vec();
        // Where in the piece, as the only store there.
        let places = [|_: usize| 0, |n: usize| n - 4, |n: usize| n / 2 / 4 * 4, |n: usize| n - 1028];
        for place in places {
            // Alternate pieces, so that untouched ones lie between.
            for odd in [0, 1] {
                let at: Vec<usize> = all.iter().skip(odd).step_by(2).map(|&(at, n)| at + place(n)).collect();
                for &at in &at {
                    title[at..at + 4].fill(1);
                    expected[at..at + 4].fill(9);
                }
                gpu_stores(host, &shadow, &at, || ());
                assert!(title == expected);
            }
        }
    }

    /// Title writes into a piece while the GPU writes it survive, and
    /// shaders see them in the next submission.
    #[test]
    fn shaders_see_what_the_title_wrote_meanwhile() {
        let Some(host) = crate::host::host() else { return };
        let Some((title, shadow)) = title_buffer() else { return };
        let mine = title.as_ptr() as usize + 3 * PIECE + 8;
        gpu_stores(host, &shadow, &[3 * PIECE, 3 * PIECE + 64], move || {
            // SAFETY: inside the title buffer.
            unsafe { (mine as *mut u8).write_volatile(7) }
        });
        let here = &title[3 * PIECE..3 * PIECE + 68];
        assert_eq!((here[0], here[8], here[9], here[64]), (9, 7, 1, 9));
        assert!(seen(host, &shadow) == title);
    }
}
