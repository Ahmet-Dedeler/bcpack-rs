// bcpack-rs by Ahmet Dedeler (https://github.com/Ahmet-Dedeler, https://ahmetdedeler.com)
// Source: https://github.com/Ahmet-Dedeler/bcpack-rs
// SPDX-License-Identifier: MIT OR Apache-2.0
// If you copy, vendor or port this file, keep this header and credit bcpack-rs in your README.
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
//! * [`format`], [`bcn`], [`bc7`]: work in progress, the block rebuild of BC3, BC4,
//!   BC5 and BC7 (rules exact, tables incomplete; see `docs/findings/bc345-bc7.md`);
//! * [`ttpf`]: RDS5 texture stores, Gfx10 detiling, DDS/PNG output;
//! * [`encode`]: a synthetic encoder used by the tests;
//! * [`tables`]: the exact model tables, generated from Microsoft's
//!   MIT-licensed `silicon-rans` Verilog.
//!
//! See `docs/BCPACK_SPEC.md` for the format and the evidence behind it.

pub mod bc1;
pub mod bc7;
pub mod bcn;
pub mod encode;
pub mod format;
pub mod rans;
pub mod tables;
#[cfg(feature = "ttpf")]
pub mod ttpf;

#[cfg(feature = "ttpf")]
mod texture;
#[cfg(feature = "ttpf")]
pub use texture::*;
