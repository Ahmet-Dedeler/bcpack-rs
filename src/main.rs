use bcpack::{
    assemble_bc1_blocks, decode_multilane_stream, extract_main_lane_record, is_legal_end_state,
    parse_main_8lane_header, BcFormat, ProbabilityModel,
};
use std::env;
use std::fs;
use std::process;

fn print_usage() {
    eprintln!(
        "Usage:
  bcpack inspect <stream.bin> [--codec <hex_id>] [--blocks <N>]
  bcpack decode-bc1 <stream.bin> <out.bc1> <num_blocks>"
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        print_usage();
        process::exit(1);
    }

    match args[1].as_str() {
        "inspect" => {
            let path = &args[2];
            let data = fs::read(path).unwrap_or_else(|e| {
                eprintln!("Failed to read {path}: {e}");
                process::exit(1);
            });
            let mut codec_id = 0x04u8;
            let mut num_blocks = 1536usize;
            let mut i = 3;
            while i + 1 < args.len() {
                match args[i].as_str() {
                    "--codec" => {
                        let s = args[i + 1].trim_start_matches("0x");
                        codec_id = u8::from_str_radix(s, 16).unwrap_or(0x04);
                        i += 2;
                    }
                    "--blocks" => {
                        num_blocks = args[i + 1].parse().unwrap_or(1536);
                        i += 2;
                    }
                    _ => i += 1,
                }
            }
            let fmt = BcFormat::from_codec_id(codec_id).unwrap_or(BcFormat::Bc1);
            let abs = fmt.lane_alphabet_bits();
            println!(
                "Stream: {} ({} bytes), format={:?}, blocks={}",
                path,
                data.len(),
                fmt,
                num_blocks
            );
            if let Some(hdrs) = parse_main_8lane_header(&data) {
                for (li, h) in hdrs.iter().enumerate() {
                    let rec = extract_main_lane_record(&data, li).unwrap();
                    let ab = abs[li];
                    let eab = ab.saturating_sub(h.z_adjust).clamp(1, 8);
                    let mut status = String::new();
                    if !h.hdr3 && h.payload_bytes == 4 {
                        let m = ProbabilityModel::new(eab, h.model_q);
                        if let Ok((syms, end_x)) = m.decode_fragment(&rec[2..6], 65536) {
                            status = format!(
                                " -> CONVERGED ({} syms, end_x={:#010x}, legal={})",
                                syms.len(),
                                end_x,
                                is_legal_end_state(end_x)
                            );
                        }
                    }
                    println!(
                        "  L{li}: mode={:#04x} (z={}, hdr3={}, flush={}, q={:2}, eab={}) len={:5}{}",
                        h.raw_mode,
                        h.z_adjust,
                        h.hdr3 as u8,
                        h.flush_per_frag as u8,
                        h.model_q,
                        eab,
                        h.payload_bytes,
                        status
                    );
                }
            }
        }
        "decode-bc1" => {
            if args.len() < 5 {
                print_usage();
                process::exit(1);
            }
            let in_path = &args[2];
            let out_path = &args[3];
            let num_blocks: usize = args[4].parse().unwrap_or_else(|_| {
                eprintln!("Invalid num_blocks");
                process::exit(1);
            });
            let data = fs::read(in_path).unwrap_or_else(|e| {
                eprintln!("Failed to read {in_path}: {e}");
                process::exit(1);
            });
            let abs = BcFormat::Bc1.lane_alphabet_bits();
            let lanes = decode_multilane_stream(&data, abs, num_blocks).unwrap_or_else(|e| {
                eprintln!("Decode error: {e}");
                process::exit(1);
            });
            let bc1 = assemble_bc1_blocks(&lanes, num_blocks).unwrap_or_else(|e| {
                eprintln!("Assembly error: {e}");
                process::exit(1);
            });
            fs::write(out_path, &bc1).unwrap_or_else(|e| {
                eprintln!("Failed to write {out_path}: {e}");
                process::exit(1);
            });
            println!("Decoded {} BC1 blocks ({} bytes) -> {}", num_blocks, bc1.len(), out_path);
        }
        _ => {
            print_usage();
            process::exit(1);
        }
    }
}
