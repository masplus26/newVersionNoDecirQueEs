//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! Only what changed is detiled again, by 64 KiB pages of the memory.
//! "Changed" means the page was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of each page at every submission
//! that samples the texture.
//!
//! shortcut: the staging buffer is copied whole into the host texture.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;
use crate::watch::PAGE;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    #[cfg(feature = "diagnostics")]
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    last: Mutex<Last>,
}

/// What a texture's staging buffer holds: the hash of each page detiled
/// (memory not watched), or the write-watch asking it was detiled after,
/// with room to read the memory into.
#[derive(Default)]
struct Last {
    hashes: Vec<blake3::Hash>,
    asked: u64,
    read: Vec<u8>,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    set.get_or_insert_with(Default::default)
        .insert(resource as usize);
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_written || !list.first_use(resource as usize) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging);
}

#[cfg(test)]
impl Staging {
    /// The state of a single-subresource texture outside any title resource.
    pub(crate) fn for_bench(
        host: &crate::host::Host,
        layout: victory_formats::texture::Layout,
        texture: &ID3D12Resource,
        memory: *const u8,
    ) -> Arc<Staging> {
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let (mut rows, mut total) = (0, 0);
        // SAFETY: outputs for one subresource of a live texture.
        unsafe {
            let d = texture.GetDesc();
            host.device.GetCopyableFootprints(
                &d,
                0,
                1,
                0,
                Some(&mut footprint),
                Some(&mut rows),
                None,
                Some(&mut total),
            );
        }
        let (buffer, data) = host.upload_buffer(total).expect("upload memory");
        Arc::new(Staging {
            buffer,
            #[cfg(feature = "diagnostics")]
            size: total,
            data,
            footprints: vec![footprint],
            rows: vec![rows],
            layout,
            mips: 1,
            memory,
            last: Mutex::default(),
        })
    }

    /// Where the host texture is copied from.
    pub(crate) fn source(&self) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&self.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: self.footprints[0],
            },
        }
    }
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some((now, written)) = crate::watch::writes(start, start + size, false) {
            let changed: Vec<_> = if last.asked == 0 {
                std::iter::once(0..size).collect()
            } else {
                written
                    .iter()
                    .filter(|&&(_, w)| w > last.asked)
                    .map(|&(page, _)| page_range(page * PAGE, start, size))
                    .collect()
            };
            if changed.is_empty() {
                return 0;
            }
            last.asked = now;
            last.hashes.clear();
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            last.read.resize(size, 0);
            let mut read = 0;
            for range in &changed {
                let to = last.read[range.clone()].as_mut_ptr();
                let from = self.memory.wrapping_add(range.start);
                crate::host::copy_mapped(to, from, range.len(), false);
                read += range.len();
            }
            self.detile(&last.read, &changed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, read as u64);
            return read;
        }
        // Not watched (any more): all of it when it is again.
        last.asked = 0;
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return 0;
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        let hashes: Vec<_> = tiled.chunks(PAGE).map(blake3::hash).collect();
        let changed: Vec<_> = hashes
            .iter()
            .enumerate()
            .filter(|&(i, hash)| last.hashes.get(i) != Some(hash))
            .map(|(i, _)| i * PAGE..((i + 1) * PAGE).min(size))
            .collect();
        if changed.is_empty() {
            return size;
        }
        last.hashes = hashes;
        self.detile(tiled, &changed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(
            crate::copystat::C::TexChanged,
            changed.iter().map(|r| r.len() as u64).sum(),
        );
        size
    }

    /// Detiles the byte ranges `changed` of `tiled` (the texture's memory)
    /// into the staging buffer.
    fn detile(&self, tiled: &[u8], changed: &[std::ops::Range<usize>]) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if let Err(e) =
                self.layout
                    .detile_changed(tiled, changed, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

/// The part of the page at address `page` inside a texture's memory
/// (`start`, `size`), as offsets.
fn page_range(page: usize, start: usize, size: usize) -> std::ops::Range<usize> {
    page.saturating_sub(start).min(size)..(page + PAGE).saturating_sub(start).min(size)
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let s = Arc::new(Staging {
        buffer,
        #[cfg(feature = "diagnostics")]
        size: total,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        last: Mutex::default(),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

    const SIDE: u32 = 512;

    fn host_texture(host: &crate::host::Host) -> ID3D12Resource {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: SIDE.into(),
            Height: SIDE,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let state = D3D12_RESOURCE_STATE_COPY_DEST;
        let mut made = None;
        // SAFETY: plain creation.
        unsafe {
            host.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut made)
                .unwrap();
        }
        made.unwrap()
    }

    /// Title memory not at a page's start, write-watched or not, kept to the
    /// end of the test process.
    fn memory(size: usize, watched: bool) -> &'static mut [u8] {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const SKEW: usize = 4096;
        let flags = MEM_RESERVE | MEM_COMMIT | if watched { MEM_WRITE_WATCH } else { 0 };
        // Whole pages, which the write watch is asked about.
        let len = (size + SKEW).next_multiple_of(PAGE);
        // SAFETY: a fresh allocation, never freed.
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), len, flags, PAGE_READWRITE).cast::<u8>();
            assert!(!p.is_null());
            std::slice::from_raw_parts_mut(p.add(SKEW), size)
        }
    }

    /// After writes anywhere, the staging buffer holds what detiling the
    /// whole memory gives, and less than that was read.
    #[test]
    fn changes_are_brought_up_to_date_in_part() {
        let Some(host) = host() else { return };
        for (watched, swizzle) in [true, false]
            .into_iter()
            .flat_map(|w| [Swizzle::SW_64KB_S, Swizzle::SW_4KB_D, Swizzle::LINEAR].map(|s| (w, s)))
        {
            let layout = Layout::new(&Desc {
                dimension: Dimension::Tex2d,
                width: SIDE,
                height: SIDE,
                depth: 1,
                mips: 1,
                element_bytes: 4,
                block: 1,
                samples: 1,
                swizzle,
            })
            .unwrap();
            let memory = memory(layout.size as usize, watched);
            for (i, b) in memory.iter_mut().enumerate() {
                *b = (i as u32).wrapping_mul(2_654_435_761).rotate_left(9) as u8;
            }
            let texture = host_texture(host);
            let staging = Staging::for_bench(host, layout.clone(), &texture, memory.as_ptr());
            let pitch = staging.footprints[0].Footprint.RowPitch as usize;
            let check = || {
                let mut whole = vec![0; pitch * SIDE as usize];
                layout.detile(memory_of(&staging), 0, 0, &mut whole, pitch, 0).unwrap();
                // SAFETY: the mapped staging buffer's one subresource.
                let held = unsafe { std::slice::from_raw_parts(staging.data, whole.len()) };
                assert!(held == whole, "watched {watched}, {swizzle:?}");
            };
            staging.refresh();
            check();
            let size = memory.len();
            for at in [0, 1, PAGE - 4097, PAGE - 4096, PAGE, size / 2, size - 1] {
                memory[at] ^= 0x5a;
                memory[size - 1 - at / 3] ^= 0xa5;
                let read = staging.refresh_now();
                assert!(!watched || (1..size).contains(&read), "{read} of {size}");
                check();
            }
            assert!(!watched || staging.refresh_now() == 0, "nothing written since");
        }
    }

    fn memory_of(staging: &Staging) -> &[u8] {
        // SAFETY: the test's memory, of the layout's size.
        unsafe { std::slice::from_raw_parts(staging.memory, staging.layout.size as usize) }
    }
}
