// bcpack-rs by Ahmet Dedeler (https://github.com/Ahmet-Dedeler, https://ahmetdedeler.com)
// Source: https://github.com/Ahmet-Dedeler/bcpack-rs
// SPDX-License-Identifier: MIT OR Apache-2.0
// If you copy, vendor or port this file, keep this header and credit bcpack-rs in your README.
//! Whole-texture decode from an RDS5 store (feature `ttpf`).

use crate::{bc1, ttpf};
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
    let a = tex
        .vstart
        .checked_add(p0.off)
        .ok_or_else(|| err("virtual start overflow".into()))?;
    let end = a
        .checked_add(p0.len)
        .ok_or_else(|| err("virtual end overflow".into()))?;
    let spans = store.locate(a, end)?;
    let cap = usize::try_from(p0.len).map_err(|_| err("mip 0 size overflow".into()))?;
    let mut gfx10 = Vec::new();
    gfx10
        .try_reserve_exact(cap)
        .map_err(|_| err("mip 0 allocation failed".into()))?;
    let (mut nbc, mut unfitted) = (0, 0);
    for (ci, f, e) in spans {
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
