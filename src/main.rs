// bcpack-rs by Ahmet Dedeler (https://github.com/Ahmet-Dedeler, https://ahmetdedeler.com)
// Source: https://github.com/Ahmet-Dedeler/bcpack-rs
// SPDX-License-Identifier: MIT OR Apache-2.0
// If you copy, vendor or port this file, keep this header and credit bcpack-rs in your README.
//! `bcpack` command line tool.

use bcpack::{bc1, decode_chunk, format::Mode, rans, ttpf};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

const USAGE: &str = "bcpack: decoder for Xbox Series BCPack texture streams

Usage:
  bcpack decode-{bc1|bc3|bc4|bc5|bc7} <chunk.bin> <blocks> <out.blocks>
      Decode one BCPack chunk (trailing zero padding allowed) to BC blocks
      in stream order (= stored Gfx10 tile order).
  bcpack symbols [--format FORMAT] <chunk.bin> <blocks>
      Print the decoded lane symbol streams, one line per lane.
  bcpack ttpf-export <store.ttpf> <texture_index> <out.dds|out.png>
      Decode and detile mip 0 to DDS (all five formats) or PNG (BC1).
  bcpack scan [--jobs N] [--every K] [--max-blocks N] [--emit DIR] <store.ttpf>...
      Decode every K-th BC1 chunk and report convergence and speed.
  bcpack bench [--format FORMAT] <chunk.bin> <blocks> [seconds]
      Single-thread decode speed on one chunk.
  FORMAT is bc1 (default), bc3, bc4, bc5 or bc7.

BC1 table options:
  --window <file.json>   preset endpoint window (default: built-in data/bc1_window.json)
  --quads <file.json>    index codebook (default: built-in data/bc1_quad_table.json)
";

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("bcpack: {msg}");
    process::exit(1);
}

struct Opts {
    pos: Vec<String>,
    format: Mode,
    window: bc1::Window,
    quads: bc1::QuadTable,
    jobs: usize,
    every: usize,
    max_blocks: usize,
    emit: Option<PathBuf>,
}

fn parse(command: &str, args: &[String]) -> Opts {
    let mut o = Opts {
        pos: vec![],
        format: Mode::Bc1,
        window: bc1::Window::default_table(),
        quads: bc1::QuadTable::default_table(),
        jobs: 6,
        every: 1,
        max_blocks: usize::MAX,
        emit: None,
    };
    let mut i = 0;
    let val = |i: usize| {
        args.get(i + 1)
            .cloned()
            .unwrap_or_else(|| die(format!("{} needs a value", args[i])))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--format" => {
                if !matches!(command, "symbols" | "bench") {
                    die("--format applies to symbols and bench");
                }
                o.format = supported_mode(&val(i));
                i += 2;
            }
            "--window" => {
                let t = std::fs::read_to_string(val(i)).unwrap_or_else(|e| die(e));
                o.window = bc1::Window::from_json(&t).unwrap_or_else(|e| die(e));
                i += 2;
            }
            "--quads" => {
                let t = std::fs::read_to_string(val(i)).unwrap_or_else(|e| die(e));
                o.quads = bc1::QuadTable::from_json(&t).unwrap_or_else(|e| die(e));
                i += 2;
            }
            "--jobs" => {
                o.jobs = val(i).parse().unwrap_or_else(|_| die("bad --jobs"));
                i += 2;
            }
            "--every" => {
                o.every = val(i)
                    .parse::<usize>()
                    .unwrap_or_else(|_| die("bad --every"))
                    .max(1);
                i += 2;
            }
            "--max-blocks" => {
                o.max_blocks = val(i).parse().unwrap_or_else(|_| die("bad --max-blocks"));
                i += 2;
            }
            "--emit" => {
                o.emit = Some(PathBuf::from(val(i)));
                i += 2;
            }
            _ => {
                o.pos.push(args[i].clone());
                i += 1;
            }
        }
    }
    o
}

fn read(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| die(format!("{path}: {e}")))
}

fn supported_mode(name: &str) -> Mode {
    Mode::from_name(name)
        .filter(|m| *m != Mode::Bc6h)
        .unwrap_or_else(|| die(format!("unsupported block format {name}")))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        print!("{USAGE}");
        process::exit(if args.is_empty() { 2 } else { 0 });
    }
    let o = parse(&args[0], &args[1..]);
    match args[0].as_str() {
        c if c.starts_with("decode-") => cmd_decode(&o, supported_mode(&c[7..])),
        "symbols" => cmd_symbols(&o),
        "ttpf-export" => cmd_export(&o),
        "scan" => cmd_scan(&o),
        "bench" => cmd_bench(&o),
        c => die(format!("unknown command {c}\n\n{USAGE}")),
    }
}

/// `<chunk> <blocks> <out>`; the older `<chunk> <out> <blocks>` order is accepted too.
fn cmd_decode(o: &Opts, mode: Mode) {
    if o.pos.len() != 3 {
        die(format!(
            "usage: bcpack decode-{} <chunk.bin> <blocks> <out.blocks>",
            mode.name()
        ));
    }
    let (blocks, out) = match o.pos[1].parse::<usize>() {
        Ok(b) => (b, &o.pos[2]),
        Err(_) => (
            o.pos[2]
                .parse()
                .unwrap_or_else(|_| die("blocks must be a number")),
            &o.pos[1],
        ),
    };
    let data = read(&o.pos[0]);
    let r = decode_chunk(mode, &data, blocks, &o.window, &o.quads).unwrap_or_else(|e| die(e));
    std::fs::write(out, &r.blocks).unwrap_or_else(|e| die(e));
    eprintln!(
        "decoded {blocks} blocks; {} references before block 0, {} blocks depend on unfitted table entries",
        r.dictionary_refs, r.unfitted_blocks
    );
}

fn cmd_symbols(o: &Opts) {
    if o.pos.len() != 2 {
        die("usage: bcpack symbols <chunk.bin> <blocks>");
    }
    let data = read(&o.pos[0]);
    let blocks = o.pos[1]
        .parse()
        .unwrap_or_else(|_| die("blocks must be a number"));
    let r = o
        .format
        .decode_symbols(&data, blocks)
        .unwrap_or_else(|e| die(e));
    eprintln!(
        "consumed {} of {} bytes in {} cycles",
        r.consumed,
        data.len(),
        r.cycles
    );
    for l in 0..r.lanes {
        let s: Vec<String> = r.lane(l).iter().map(|v| v.to_string()).collect();
        println!("{}", s.join(" "));
    }
}

fn cmd_export(o: &Opts) {
    if o.pos.len() != 3 {
        die("usage: bcpack ttpf-export <store.ttpf> <texture_index> <out.dds|out.png>");
    }
    let store = ttpf::Store::open(Path::new(&o.pos[0])).unwrap_or_else(|e| die(e));
    let ti: usize = o.pos[1]
        .parse()
        .unwrap_or_else(|_| die("texture index must be a number"));
    let texs = store.textures().unwrap_or_else(|e| die(e));
    let tex = texs
        .get(ti)
        .unwrap_or_else(|| die(format!("store has {} textures", texs.len())));
    let out = &o.pos[2];
    let png = out.to_ascii_lowercase().ends_with(".png");
    if png && !bcpack::BC1_DXGI.contains(&tex.dxgi) {
        die("PNG export requires BC1; use .dds for other supported formats");
    }
    let m = bcpack::texture_mip0(&store, tex, &o.window, &o.quads).unwrap_or_else(|e| die(e));
    let bytes = if png {
        let rgba = ttpf::bc1_to_rgba(&m.blocks, m.bw, m.bh, m.width as usize, m.height as usize);
        ttpf::png_rgba(&rgba, m.width, m.height).unwrap_or_else(|e| die(e))
    } else {
        ttpf::dds(&m.blocks, m.width, m.height, tex.dxgi).unwrap_or_else(|e| die(e))
    };
    std::fs::write(out, bytes).unwrap_or_else(|e| die(e));
    eprintln!(
        "texture {ti}: {}x{}, {} BCPack chunk(s), {} blocks depend on unfitted table entries",
        m.width, m.height, m.bcpack_chunks, m.unfitted_blocks
    );
}

#[derive(Default, Clone)]
struct Tally {
    chunks: usize,
    ok: usize,
    blocks: u64,
    in_bytes: u64,
    exact_len: usize,
    clean: usize,
    errors: BTreeMap<String, usize>,
    fail_by_blocks: BTreeMap<usize, usize>,
    secs: f64,
}

impl Tally {
    fn add(&mut self, t: &Tally) {
        self.chunks += t.chunks;
        self.ok += t.ok;
        self.blocks += t.blocks;
        self.in_bytes += t.in_bytes;
        self.exact_len += t.exact_len;
        self.clean += t.clean;
        self.secs += t.secs;
        for (k, v) in &t.errors {
            *self.errors.entry(k.clone()).or_default() += v;
        }
        for (k, v) in &t.fail_by_blocks {
            *self.fail_by_blocks.entry(*k).or_default() += v;
        }
    }
}

fn cmd_scan(o: &Opts) {
    if o.pos.is_empty() {
        die("usage: bcpack scan [--jobs N] [--every K] <store.ttpf>...");
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(o.jobs)
        .build_global()
        .ok();
    if let Some(d) = &o.emit {
        std::fs::create_dir_all(d).unwrap_or_else(|e| die(e));
    }
    let t0 = Instant::now();
    let mut total = Tally::default();
    let mut stem_counts: BTreeMap<String, usize> = BTreeMap::new();
    for path in &o.pos {
        let s = Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        *stem_counts.entry(s).or_default() += 1;
    }
    for (store_idx, path) in o.pos.iter().enumerate() {
        let store = match ttpf::Store::open(Path::new(path)) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{path}: {e}");
                continue;
            }
        };
        let ids: Vec<usize> = store
            .chunks
            .iter()
            .filter(|c| {
                c.codec == ttpf::CODEC_BC1 && c.vlen >= 8 && (c.vlen / 8) as usize <= o.max_blocks
            })
            .map(|c| c.index)
            .step_by(o.every)
            .collect();
        let raw_stem = Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let stem = if stem_counts.get(&raw_stem).copied().unwrap_or(0) > 1 {
            format!("{raw_stem}_{store_idx}")
        } else {
            raw_stem
        };
        let tallies: Vec<Tally> = ids
            .par_iter()
            .map(|&ci| {
                let c = &store.chunks[ci];
                let blocks = (c.vlen / 8) as usize;
                let mut t = Tally {
                    chunks: 1,
                    ..Default::default()
                };
                let raw = match store.chunk_bytes(ci, true) {
                    Ok(r) => r,
                    Err(e) => {
                        *t.errors.entry(format!("read: {e}")).or_default() += 1;
                        return t;
                    }
                };
                let s = Instant::now();
                let r = bc1::decode_symbols(raw, blocks);
                match r {
                    Ok(lanes) => {
                        let out = bc1::assemble(&lanes, &o.window, &o.quads);
                        t.secs = s.elapsed().as_secs_f64();
                        t.ok = 1;
                        t.blocks = blocks as u64;
                        t.in_bytes = lanes.consumed as u64;
                        t.exact_len = (lanes.consumed as u64 == c.data_len()) as usize;
                        t.clean = (out.unfitted_blocks == 0) as usize;
                        if let Some(d) = &o.emit {
                            let p = d.join(format!("{stem}_{ci}.bc1"));
                            if let Err(e) = std::fs::write(&p, &out.blocks) {
                                eprintln!("{}: {e}", p.display());
                                *t.errors.entry(format!("emit: {e}")).or_default() += 1;
                            }
                        }
                    }
                    Err(e) => {
                        let key = match e {
                            rans::DecodeError::BadHeader { .. } => "bad header",
                            rans::DecodeError::NoProgress { .. } => "no progress",
                            rans::DecodeError::CycleLimit => "cycle limit",
                            rans::DecodeError::IllegalEndState { .. } => "illegal end state",
                            rans::DecodeError::TrailingData { .. } => "trailing data",
                            rans::DecodeError::Invalid(_) => "invalid",
                        };
                        *t.errors.entry(key.into()).or_default() += 1;
                        *t.fail_by_blocks.entry(blocks).or_default() += 1;
                    }
                }
                t
            })
            .collect();
        let mut st = Tally::default();
        for t in &tallies {
            st.add(t);
        }
        println!(
            "{path}: {}/{} converged, {} blocks, consumed == data length in {}, chunks with every block fitted {}",
            st.ok, st.chunks, st.blocks, st.exact_len, st.clean
        );
        total.add(&st);
    }
    let wall = t0.elapsed().as_secs_f64();
    println!(
        "TOTAL {}/{} converged ({} failed); {} blocks; consumed == data length in {}/{}; \
         chunks touching no unfitted window slot {}/{}",
        total.ok,
        total.chunks,
        total.chunks - total.ok,
        total.blocks,
        total.exact_len,
        total.ok,
        total.clean,
        total.ok
    );
    if !total.errors.is_empty() {
        println!("errors: {:?}", total.errors);
        println!("failures by block count: {:?}", total.fail_by_blocks);
    }
    if total.secs > 0.0 {
        println!(
            "decode wall time summed over chunks {:.2} s: {:.1} MB/s compressed in, {:.1} MB/s BC1 out per thread; total wall {:.1} s (incl. reading stores)",
            total.secs,
            total.in_bytes as f64 / total.secs / 1e6,
            total.blocks as f64 * 8.0 / total.secs / 1e6,
            wall
        );
    }
}

/// CPU time of the calling thread in seconds when supported by the OS (the
/// machine may be shared, so wall time can understate the decoder's speed).
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn thread_cpu_secs() -> Option<f64> {
    #[repr(C)]
    struct Timespec {
        sec: std::ffi::c_long,
        nsec: std::ffi::c_long,
    }
    extern "C" {
        fn clock_gettime(clk: std::ffi::c_int, tp: *mut Timespec) -> std::ffi::c_int;
    }
    #[cfg(target_os = "macos")]
    const CLOCK_THREAD_CPUTIME_ID: std::ffi::c_int = 16;
    #[cfg(target_os = "linux")]
    const CLOCK_THREAD_CPUTIME_ID: std::ffi::c_int = 3;
    let mut t = Timespec { sec: 0, nsec: 0 };
    if unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mut t) } == 0 {
        Some(t.sec as f64 + t.nsec as f64 * 1e-9)
    } else {
        None
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn thread_cpu_secs() -> Option<f64> {
    None
}

fn cmd_bench(o: &Opts) {
    if o.pos.len() < 2 {
        die("usage: bcpack bench <chunk.bin> <blocks> [seconds]");
    }
    let data = read(&o.pos[0]);
    let blocks: usize = o.pos[1]
        .parse()
        .unwrap_or_else(|_| die("blocks must be a number"));
    let secs: f64 = match o.pos.get(2) {
        Some(raw) => {
            let v: f64 = raw
                .parse()
                .unwrap_or_else(|_| die("seconds must be a positive number"));
            if !v.is_finite() || v <= 0.0 {
                die("seconds must be a positive number");
            }
            v
        }
        None => 2.0,
    };
    let decode =
        || decode_chunk(o.format, &data, blocks, &o.window, &o.quads).unwrap_or_else(|e| die(e));
    let warmup = decode();
    let (consumed, cycles) = (warmup.consumed, warmup.cycles);
    drop(warmup);
    let t0 = Instant::now();
    let c0 = thread_cpu_secs();
    let mut n = 0u64;
    while n == 0 || t0.elapsed().as_secs_f64() < secs {
        std::hint::black_box(decode());
        n += 1;
    }
    let wall = t0.elapsed().as_secs_f64();
    let cpu = c0.and_then(|start| thread_cpu_secs().map(|end| end - start));
    let s = cpu.filter(|&d| d > 0.0).unwrap_or(wall).max(1e-9);
    println!(
        "{n} decodes, {s:.2} s CPU ({wall:.2} s wall): {:.1} MB/s compressed in, {:.1} MB/s {} out, \
         {:.2} Mblocks/s per core; {:.2} clocks per block",
        (n as f64 * consumed as f64) / s / 1e6,
        (n as f64 * blocks as f64 * o.format.block_bytes() as f64) / s / 1e6,
        o.format.name().to_ascii_uppercase(),
        (n as f64 * blocks as f64) / s / 1e6,
        cycles as f64 / blocks as f64
    );
}
