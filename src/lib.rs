//! Decoder for Xbox Series BCPack BC1 texture streams.
//!
//! BCPack is the rANS entropy layer the Xbox Series MDU puts over BCn blocks.
//! This crate decodes the BC1 variant (codec id 0x04) to plain BC1 blocks:
//!
//! ```no_run
//! let chunk = std::fs::read("chunk.bin").unwrap();
//! let window = bcpack::bc1::Window::default_table();
//! let quads = bcpack::bc1::QuadTable::default_table();
//! let out = bcpack::bc1::decode(&chunk, 4096, &window, &quads).unwrap();
//! assert_eq!(out.blocks.len(), 4096 * 8);
//! ```
//!
//! Modules:
//! * [`rans`]: the 12-lane cycle-level rANS decoder (entropy layer);
//! * [`bc1`]: lane symbols -> BC1 endpoints and index words;
//! * [`ttpf`]: RDS5 texture stores, Gfx10 detiling, DDS/PNG output;
//! * [`encode`]: a synthetic encoder used by the tests;
//! * [`tables`]: the exact model tables, generated from Microsoft's
//!   MIT-licensed `silicon-rans` Verilog.
//!
//! See `docs/BCPACK_SPEC.md` for the format and the evidence behind it.

pub mod bc1;
pub mod encode;
pub mod rans;
pub mod tables;
pub mod ttpf;

use std::io;

/// Mip 0 of a BC1 texture, decoded and detiled.
pub struct Mip0 {
    /// Row-major BC1 blocks, `bw * bh * 8` bytes.
    pub blocks: Vec<u8>,
    pub bw: usize,
    pub bh: usize,
    pub width: u32,
    pub height: u32,
    /// Mip 0 exactly as stored (Gfx10 tiled, padded to whole tiles).
    pub gfx10: Vec<u8>,
    /// BCPack chunks decoded / blocks depending on unfitted window slots.
    pub bcpack_chunks: usize,
    pub unfitted_blocks: usize,
}

/// DXGI formats this exporter handles (BC1_TYPELESS, BC1_UNORM, BC1_UNORM_SRGB).
pub const BC1_DXGI: [u32; 3] = [70, 71, 72];

/// Decode mip 0 of texture `tex` in `store`: plain chunks are inflated, BCPack
/// BC1 chunks are decoded whole and the covering slice is used.
pub fn texture_mip0(
    store: &ttpf::Store,
    tex: &ttpf::Texture,
    window: &bc1::Window,
    quads: &bc1::QuadTable,
) -> io::Result<Mip0> {
    let err = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
    if !BC1_DXGI.contains(&tex.dxgi) || tex.array_size != 1 || tex.tile_mode > 3 {
        return Err(err(format!(
            "texture {}: only single BC1 textures are supported (dxgi {}, array {}, tile mode {})",
            tex.index, tex.dxgi, tex.array_size, tex.tile_mode
        )));
    }
    let (levels, _) = ttpf::mip_chain_bc1(tex.tile_mode, tex.width, tex.height, tex.mips)
        .ok_or_else(|| err("unsupported mip layout".into()))?;
    let p0 = &levels[0];
    let a = tex.vstart + p0.off;
    let mut gfx10 = Vec::with_capacity(p0.len as usize);
    let (mut nbc, mut unfitted) = (0, 0);
    for (ci, f, e) in store.locate(a, a + p0.len)? {
        let c = &store.chunks[ci];
        let bytes = match c.codec {
            ttpf::CODEC_BC1 => {
                let raw = store.chunk_bytes(ci, true)?;
                let out = bc1::decode(raw, (c.vlen / 8) as usize, window, quads)
                    .map_err(|e| err(format!("chunk {ci}: {e}")))?;
                nbc += 1;
                unfitted += out.unfitted_blocks;
                out.blocks
            }
            ttpf::CODEC_ZLIB | ttpf::CODEC_STORED => store.plain_chunk(ci)?,
            x => return Err(err(format!("chunk {ci}: codec {x:#x} not supported"))),
        };
        gfx10.extend_from_slice(&bytes[f as usize..e as usize]);
    }
    let (blocks, bw, bh) = ttpf::detile_bc1(&gfx10, p0, tex.tile_mode);
    Ok(Mip0 {
        blocks,
        bw,
        bh,
        width: p0.w,
        height: p0.h,
        gfx10,
        bcpack_chunks: nbc,
        unfitted_blocks: unfitted,
    })
}
