//! A synthetic world-like load through the real submission path
//! ([`Host::execute`]): a big title heap that shaders read and write through
//! overlapping views, indirect draws fed by those writes, textures in title
//! memory, and title writes scattered over the heap each frame. Uses no
//! game data. `cargo gpu-bench` runs it:
//!
//! * `profile` prints where the time of a frame goes;
//! * `matches_the_software_adapter` draws a smaller scene on the GPU and on
//!   WARP (its own process) and compares the image and title memory.

use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{PCSTR, s};

use crate::host::{Host, HostList, host, transition};
use crate::shadow::Shadow;
use crate::texture::Staging;
use crate::watch::PAGE;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// Pages at the heap's start: the indirect arguments, then room.
const HEADER: usize = 4;
/// Bytes between the dwords a dispatch changes.
const STRIDE: usize = 1024;
/// Quads an indirect draw makes, one per changed dword.
const QUADS: u32 = 4096;
/// The part of a texture shown in the image.
const SHOWN: u32 = 128;
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// What a frame does.
#[derive(Clone, Copy)]
struct Scene {
    heap: usize,
    submissions: usize,
    /// Per submission: overlapping views, each written by a dispatch.
    views: usize,
    /// Dwords each dispatch changes.
    writes: u32,
    /// Heap pages the title writes between submissions.
    title_pages: usize,
    texture: u32,
}

const WORLD: Scene = Scene {
    heap: 174 << 20,
    submissions: 6,
    views: 24,
    writes: 8192,
    title_pages: 32,
    texture: 2048,
};

const SMALL: Scene = Scene {
    heap: 16 << 20,
    submissions: 2,
    views: 4,
    writes: QUADS,
    title_pages: 8,
    texture: 256,
};

const ROOT: &str = "RootFlags(0), RootConstants(num32BitConstants=4, b0), UAV(u0), SRV(t0)";

const SOURCE: &str = r#"
cbuffer C : register(b0) { uint first, count, stride, frame; };
RWByteAddressBuffer target : register(u0);
ByteAddressBuffer heap : register(t0);

[numthreads(64, 1, 1)]
void change(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    uint at = first + id.x * stride;
    target.Store(at, target.Load(at) * 1664525 + 1013904223 + frame + id.x);
}

struct V { float4 at : SV_Position; nointerpolation uint color : COLOR; };

V quad(uint v : SV_VertexID, uint i : SV_InstanceID) {
    static const uint2 CORNERS[6] = { {0,0}, {1,0}, {0,1}, {0,1}, {1,0}, {1,1} };
    uint2 cell = uint2(i % 80, i / 80 % 45);
    uint2 pixel = (cell + CORNERS[v]) * 16;
    V o;
    o.at = float4(pixel.x / 640.0 - 1, 1 - pixel.y / 360.0, 0, 1);
    o.color = heap.Load(first + i * stride);
    return o;
}

float4 paint(V v) : SV_Target {
    return float4(uint4(v.color, v.color >> 8, v.color >> 16, v.color >> 24) & 255) / 255.0;
}
"#;

fn compile(entry: PCSTR, target: PCSTR) -> ID3DBlob {
    let (mut code, mut errors) = (None, None);
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("bench"), None, None, entry, target, 0, 0, &mut code, Some(&mut errors))
    };
    if let Err(e) = compiled {
        let text = errors.map(|b: ID3DBlob| String::from_utf8_lossy(bytes(&b)).into_owned());
        panic!("{e}: {text:?}");
    }
    code.unwrap()
}

fn bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: the blob's own bytes, borrowed with it.
    unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) }
}

fn code(blob: &ID3DBlob) -> D3D12_SHADER_BYTECODE {
    let b = bytes(blob);
    D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr().cast(), BytecodeLength: b.len() }
}

/// Title memory for the life of the process, write-watched or not.
fn title_memory(len: usize, watched: bool) -> &'static mut [u8] {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    let flags = MEM_RESERVE | MEM_COMMIT | if watched { MEM_WRITE_WATCH } else { 0 };
    // SAFETY: a fresh allocation, never freed.
    unsafe {
        let p = VirtualAlloc(std::ptr::null(), len, flags, PAGE_READWRITE).cast::<u8>();
        assert!(!p.is_null());
        std::slice::from_raw_parts_mut(p, len)
    }
}

fn texture(host: &Host, width: u32, height: u32, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width.into(),
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Flags: flags,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let mut made = None;
    // SAFETY: plain creation.
    unsafe { host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut made) }.unwrap();
    made.unwrap()
}

fn at_subresource(texture: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        ..Default::default()
    }
}

/// A title texture and its host copy.
struct Streamed {
    memory: &'static mut [u8],
    staging: Arc<Staging>,
    copy: ID3D12Resource,
}

impl Streamed {
    fn new(host: &Host, side: u32, watched: bool) -> Streamed {
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: side,
            height: side,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        })
        .unwrap();
        let memory = title_memory(layout.size as usize, watched);
        for (i, b) in memory.iter_mut().enumerate() {
            *b = (i as u32).wrapping_mul(2_654_435_761).rotate_left(9) as u8;
        }
        let copy = texture(host, side, side, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        let staging = Staging::for_bench(host, layout, &copy, memory.as_ptr());
        Streamed { memory, staging, copy }
    }
}

struct World {
    scene: Scene,
    host: &'static Host,
    heap: &'static mut [u8],
    shadow: Arc<Shadow>,
    textures: [Streamed; 2],
    target: ID3D12Resource,
    rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    _rtvs: ID3D12DescriptorHeap,
    image: (ID3D12Resource, *mut u8),
    root: ID3D12RootSignature,
    change: ID3D12PipelineState,
    draw: ID3D12PipelineState,
    indirect: ID3D12CommandSignature,
    lists: Vec<HostList>,
}

impl World {
    fn new(scene: Scene) -> World {
        let host = host().expect("a D3D12 device");
        let heap = title_memory(scene.heap, true);
        for (i, dword) in heap.chunks_exact_mut(4).enumerate() {
            dword.copy_from_slice(&(i as u32).wrapping_mul(2_654_435_761).to_le_bytes());
        }
        // Indirect arguments: vertices, instances, first vertex and instance.
        for (i, v) in [6u32, QUADS, 0, 0].into_iter().enumerate() {
            heap[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        let start = heap.as_ptr() as u64;
        crate::shadow::register(start as usize, start, heap.len() as u64);
        let shadow = crate::shadow::find(start, heap.len() as u64).expect("a host copy");
        let target = texture(host, WIDTH, HEIGHT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET);
        let (root, change, draw, indirect) = pipelines(&host.device);
        // SAFETY: plain creation on live objects.
        let (rtvs, rtv) = unsafe {
            let rtvs: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .unwrap();
            let rtv = rtvs.GetCPUDescriptorHandleForHeapStart();
            host.device.CreateRenderTargetView(&target, None, rtv);
            (rtvs, rtv)
        };
        World {
            scene,
            host,
            heap,
            shadow,
            textures: [true, false].map(|watched| Streamed::new(host, scene.texture, watched)),
            target,
            rtv,
            _rtvs: rtvs,
            image: host.shared_buffer(u64::from(WIDTH * HEIGHT * 4)).unwrap(),
            root,
            change,
            draw,
            indirect,
            lists: (0..scene.submissions)
                .map(|_| {
                    let list = host.new_list().unwrap();
                    list.close();
                    list
                })
                .collect(),
        }
    }

    /// The title's own writes before a submission (never the dwords shaders
    /// change): one per page, scattered, and a texel of the watched texture.
    fn title_writes(&mut self, frame: u32, submission: usize) {
        let pages = self.scene.heap / PAGE - HEADER;
        for i in 0..self.scene.title_pages {
            let n = frame as usize * self.scene.submissions + submission;
            let page = HEADER + (n * self.scene.title_pages + i) * 97 % pages;
            self.heap[page * PAGE + STRIDE / 2] = (frame + i as u32) as u8;
        }
        if submission == 0 {
            let memory = &mut self.textures[0].memory;
            let at = frame as usize * 8191 % memory.len();
            memory[at] = !memory[at];
        }
    }

    fn frame(&mut self, frame: u32) {
        for submission in 0..self.scene.submissions {
            self.title_writes(frame, submission);
            let list = &self.lists[submission];
            list.reset();
            self.record(list, frame, submission);
            list.close();
            self.host.execute(&[list]);
        }
    }

    fn record(&self, list: &HostList, frame: u32, submission: usize) {
        let start = self.heap.as_ptr() as u64;
        let base = |view: usize| (HEADER + view) * PAGE;
        let l = &list.list;
        self.shadow.refresh(list, start, PAGE as u64);
        // SAFETY: an open list and live objects, kept past its execution.
        unsafe {
            l.SetComputeRootSignature(&self.root);
            for view in 0..self.scene.views {
                let (from, len) = (start + base(view) as u64, (self.scene.heap - base(view) - view * PAGE) as u64);
                self.shadow.refresh(list, from, len);
                self.shadow.gpu_writes(list, from, len);
                let constants = [0, self.scene.writes, STRIDE as u32, frame];
                l.SetPipelineState(&self.change);
                l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                l.SetComputeRootUnorderedAccessView(1, self.shadow.gpu(from));
                l.Dispatch(self.scene.writes.div_ceil(64), 1, 1);
                l.ResourceBarrier(&[crate::indirect::uav_barrier()]);
            }
            if submission == 0 {
                l.ClearRenderTargetView(self.rtv, &[0.0; 4], None);
            }
            self.record_draw(list, start + base(submission % self.scene.views) as u64);
            if submission == 0 {
                self.record_textures(list);
            }
        }
        // The title writes pages the dispatches write while the GPU has
        // them: after the copies were refreshed and their ranges declared.
        for i in 0..self.scene.title_pages {
            let page = HEADER + self.scene.views + i * 13 % (self.scene.heap / PAGE / 16);
            let at = start as usize + page * PAGE + 3 * STRIDE / 4;
            // SAFETY: inside the heap, which lives as long as the process.
            list.defer(move || unsafe { (at as *mut u8).write_volatile((frame + i as u32) as u8) });
        }
    }

    /// Quads coloured by what the dispatches wrote, through the clamp the
    /// title's indirect draws go through.
    fn record_draw(&self, list: &HostList, colors: u64) {
        let l = &list.list;
        let args = self.shadow.gpu(self.heap.as_ptr() as u64);
        assert!(crate::indirect::bound(list, &self.shadow.buffer, args, (0, 16, 1)));
        // SAFETY: an open list and live objects.
        unsafe {
            let readable = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            let both = [
                transition(&self.shadow.buffer, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, readable),
                transition(&self.shadow.buffer, readable, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
            ];
            l.ResourceBarrier(&both[..1]);
            l.SetGraphicsRootSignature(&self.root);
            l.SetPipelineState(&self.draw);
            let constants = [0, 0, STRIDE as u32, 0];
            l.SetGraphicsRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            l.SetGraphicsRootShaderResourceView(2, self.shadow.gpu(colors));
            l.OMSetRenderTargets(1, Some(&self.rtv), false, None);
            l.RSSetViewports(&[D3D12_VIEWPORT { Width: WIDTH as f32, Height: HEIGHT as f32, MaxDepth: 1.0, ..Default::default() }]);
            l.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: WIDTH as i32, bottom: HEIGHT as i32 }]);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            l.ExecuteIndirect(&self.indirect, 1, &self.shadow.buffer, 0, None, 0);
            l.ResourceBarrier(&both[1..]);
        }
        crate::indirect::done(list, &self.shadow.buffer);
    }

    fn record_textures(&self, list: &HostList) {
        let (copy, source) = (D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        let corner = D3D12_BOX { right: SHOWN, bottom: SHOWN, back: 1, ..Default::default() };
        let l = &list.list;
        // SAFETY: an open list and live objects.
        unsafe {
            l.ResourceBarrier(&[transition(&self.target, D3D12_RESOURCE_STATE_RENDER_TARGET, copy)]);
            for (i, t) in self.textures.iter().enumerate() {
                list.defer_texture(t.staging.clone());
                l.CopyTextureRegion(&at_subresource(&t.copy), 0, 0, 0, &t.staging.source(), None);
                l.ResourceBarrier(&[transition(&t.copy, copy, source)]);
                let to = at_subresource(&self.target);
                l.CopyTextureRegion(&to, i as u32 * SHOWN, 0, 0, &at_subresource(&t.copy), Some(&corner));
                l.ResourceBarrier(&[transition(&t.copy, source, copy)]);
            }
            l.ResourceBarrier(&[transition(&self.target, copy, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
        }
    }

    /// The title releases the heap.
    fn leave(self) {
        let start = self.heap.as_ptr() as u64;
        crate::shadow::unregister(start as usize, start);
    }

    /// The image as drawn, the textures as uploaded and title memory.
    fn result(&self) -> Vec<u8> {
        let list = &self.lists[0];
        let source = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&self.image.0) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: WIDTH,
                        Height: HEIGHT,
                        Depth: 1,
                        RowPitch: WIDTH * 4,
                    },
                },
            },
        };
        list.reset();
        // The textures over the quads, so that the image shows both.
        self.record_textures(list);
        // SAFETY: an open list and live objects; the mapping after the copy.
        unsafe {
            let l = &list.list;
            l.ResourceBarrier(&[transition(&self.target, D3D12_RESOURCE_STATE_RENDER_TARGET, source)]);
            l.CopyTextureRegion(&to, 0, 0, 0, &at_subresource(&self.target), None);
            l.ResourceBarrier(&[transition(&self.target, source, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
            list.close();
            self.host.execute(&[list]);
            let mut result = std::slice::from_raw_parts(self.image.1, (WIDTH * HEIGHT * 4) as usize).to_vec();
            result.extend_from_slice(self.heap);
            result
        }
    }
}

fn pipelines(device: &ID3D12Device) -> (ID3D12RootSignature, ID3D12PipelineState, ID3D12PipelineState, ID3D12CommandSignature) {
    let (cs, vs, ps) = (
        compile(s!("change"), s!("cs_5_1")),
        compile(s!("quad"), s!("vs_5_1")),
        compile(s!("paint"), s!("ps_5_1")),
    );
    // SAFETY: plain creation; blobs and the root signature outlive the calls.
    unsafe {
        let (mut blob, mut errors) = (None, None);
        let text = format!("#define R \"{ROOT}\"\0");
        D3DCompile(text.as_ptr().cast(), text.len() - 1, None, None, None, s!("R"), s!("rootsig_1_1"), 0, 0, &mut blob, Some(&mut errors)).unwrap();
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes(&blob.unwrap())).unwrap();
        let change = device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: code(&cs),
                ..Default::default()
            })
            .unwrap();
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            VS: code(&vs),
            PS: code(&ps),
            SampleMask: u32::MAX,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                ..Default::default()
            },
            ..Default::default()
        };
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 15;
        let draw = device.CreateGraphicsPipelineState(&desc).unwrap();
        let argument = [D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, ..Default::default() }];
        let mut indirect: Option<ID3D12CommandSignature> = None;
        device
            .CreateCommandSignature(
                &D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 16, NumArgumentDescs: 1, pArgumentDescs: argument.as_ptr(), ..Default::default() },
                None,
                &mut indirect,
            )
            .unwrap();
        (root, change, draw, indirect.unwrap())
    }
}

/// The host adapter's name.
fn adapter(host: &Host) -> String {
    use windows::Win32::Graphics::Dxgi::*;
    // SAFETY: plain queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(host.device.GetAdapterLuid()).unwrap();
        let name = adapter.GetDesc1().unwrap().Description;
        String::from_utf16_lossy(&name).trim_end_matches('\0').to_owned()
    }
}

/// Milliseconds and megabytes since the last call: preparing, of it buffer
/// copies and textures, GPU, copying back; copied, read for textures and
/// written back.
fn counters() -> [f64; 8] {
    use crate::host::{COPY_BACK, PREPARING, SUBMISSION_US};
    let ms = |c: &std::sync::atomic::AtomicU64| c.swap(0, Relaxed) as f64 / 1e3;
    let mb = |c: &std::sync::atomic::AtomicU64| c.swap(0, Relaxed) as f64 / f64::from(1 << 20);
    [
        ms(&SUBMISSION_US[0]),
        ms(&PREPARING[0]),
        ms(&PREPARING[2]),
        ms(&SUBMISSION_US[1]),
        ms(&COPY_BACK[0]),
        mb(&PREPARING[1]),
        mb(&PREPARING[3]),
        mb(&COPY_BACK[1]),
    ]
}

/// The first quartile: what else the machine does only makes frames slower,
/// often for seconds at a time.
fn typical(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 4]
}

#[test]
#[ignore = "a benchmark: cargo gpu-bench"]
fn profile() {
    const WARM_UP: u32 = 20;
    const FRAMES: u32 = 300;
    const NAMES: [&str; 8] = [
        "  prepare          ms", "    buffer copies  ms", "    textures       ms", "  GPU              ms",
        "  copy back        ms", "buffers copied     MB", "textures read      MB", "written back       MB",
    ];
    let mut world = World::new(WORLD);
    eprintln!("{}: {} submissions of {} overlapping views of {} MiB", adapter(world.host), WORLD.submissions, WORLD.views, WORLD.heap >> 20);
    (0..WARM_UP).for_each(|f| world.frame(f));
    let mut frames = Vec::new();
    let mut phases = vec![Vec::new(); NAMES.len()];
    for frame in WARM_UP..WARM_UP + FRAMES {
        counters();
        let started = Instant::now();
        world.frame(frame);
        frames.push(started.elapsed().as_secs_f64() * 1e3);
        for (all, value) in phases.iter_mut().zip(counters()) {
            all.push(value);
        }
    }
    let typical_frame = typical(&mut frames);
    eprintln!("frame {typical_frame:7.2} ms (median {:.2}, best {:.2})", frames[frames.len() / 2], frames[0]);
    for (name, values) in NAMES.iter().zip(&mut phases) {
        eprintln!("{name} {:8.2}", typical(values));
    }
}

/// The first frames of a scene whose heap and textures are new, after one
/// that has settled.
#[test]
#[ignore = "a benchmark: cargo gpu-bench"]
fn scene_change() {
    const FRAMES: u32 = 6;
    let mut settled = World::new(SMALL);
    (0..30).for_each(|f| settled.frame(f));
    for round in 0..3 {
        counters();
        let started = Instant::now();
        let mut world = World::new(WORLD);
        eprintln!("scene {round}: made in {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
        for frame in 0..FRAMES {
            counters();
            let started = Instant::now();
            world.frame(frame);
            let ms = started.elapsed().as_secs_f64() * 1e3;
            let [prepare, copies, textures, gpu, back, ..] = counters();
            eprintln!(
                "  frame {frame}: {ms:7.1} ms (prepare {prepare:.1}: copies {copies:.1}, textures {textures:.1}; GPU {gpu:.1}; back {back:.1})"
            );
        }
        world.leave();
    }
}

/// Set in the child process that draws on WARP: where the result goes.
const REFERENCE: &str = "VICTORY_BENCH_REFERENCE";

#[test]
#[ignore = "starts a second process: cargo gpu-bench"]
fn matches_the_software_adapter() {
    const FRAMES: u32 = 4;
    let mut world = World::new(SMALL);
    (0..FRAMES).for_each(|f| world.frame(f));
    let result = world.result();
    if let Some(path) = std::env::var_os(REFERENCE) {
        std::fs::write(path, result).unwrap();
        return;
    }
    let name = adapter(world.host);
    let kept = if world.shadow.vram().is_some() { "video" } else { "system" };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reference");
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "bench::matches_the_software_adapter", "--ignored"])
        .env(REFERENCE, &path)
        .env("VICTORY_D3D12_WARP", "1")
        .output()
        .unwrap();
    assert!(child.status.success(), "{}", String::from_utf8_lossy(&child.stdout));
    let reference = std::fs::read(path).unwrap();
    let (image, heap) = result.split_at((WIDTH * HEIGHT * 4) as usize);
    let (image_warp, heap_warp) = reference.split_at(image.len());
    let colors: std::collections::HashSet<&[u8]> = image.chunks_exact(4).collect();
    assert!(colors.len() > 1000, "only {} colours drawn", colors.len());
    let differ = |a: &[u8], b: &[u8]| a.iter().zip(b).filter(|(a, b)| a != b).count();
    assert_eq!(differ(image, image_warp), 0, "image bytes");
    assert_eq!(heap.len(), heap_warp.len());
    assert_eq!(differ(heap, heap_warp), 0, "title memory bytes");
    eprintln!("{name} ({kept} memory) and WARP agree: {} colours, {} bytes of title memory", colors.len(), heap.len());
}
