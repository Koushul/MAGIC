//! Wall-clock comparison: legacy Rust vs optimized Rust vs optional Python impute-only.
//!
//! Usage:
//!   cargo run --release -p magic-impute --bin compare_magic_timings -- --help

use magic_impute::{impute_magic, impute_magic_f32, impute_magic_legacy, synthetic, ImputeConfig};
use std::process::Command;
use std::time::Instant;

fn median_sorted(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn main() {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut run_python = false;
    args.retain(|a| {
        if a == "--python" {
            run_python = true;
            false
        } else {
            true
        }
    });

    let n = args
        .iter()
        .position(|s| s == "--n")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048usize);
    let p = args
        .iter()
        .position(|s| s == "--p")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(512usize);
    let knn = args
        .iter()
        .position(|s| s == "--knn")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(32usize);
    let t = args
        .iter()
        .position(|s| s == "--t")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(3u32);
    let threads = args
        .iter()
        .position(|s| s == "--threads")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8usize);
    let repeats = args
        .iter()
        .position(|s| s == "--repeat")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(15usize);
    let block = args
        .iter()
        .position(|s| s == "--block")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(128usize);

    let csr = synthetic::csr_row_stochastic(n, knn, 42);
    let x = synthetic::dense_expression(n, p, 1);

    let cfg = ImputeConfig {
        threads: Some(threads),
        gene_block_size: block,
    };

    let mut legacy_ms = Vec::with_capacity(repeats);
    let mut opt_ms = Vec::with_capacity(repeats);
    let mut f32_ms = Vec::with_capacity(repeats);

    for _ in 0..repeats {
        let t0 = Instant::now();
        let _ = impute_magic_legacy(&csr, &x, t, Some(threads));
        legacy_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

        let t0 = Instant::now();
        let _ = impute_magic(&csr, &x, t, &cfg);
        opt_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

        let x32 = x.mapv(|v| v as f32);
        let t0 = Instant::now();
        let _ = impute_magic_f32(&csr, &x32, t, &cfg);
        f32_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let leg = median_sorted(&mut legacy_ms);
    let opt = median_sorted(&mut opt_ms);
    let f32t = median_sorted(&mut f32_ms);

    println!(
        "Synthetic n={} p={} knn={} diffusion_t={} threads={} block={} repeats={}",
        n, p, knn, t, threads, block, repeats
    );
    println!(
        "Rust legacy (row×full-gene SpMM, alloc/iter): median {:.3} ms",
        leg
    );
    println!(
        "Rust optimized (blocked SpMM, double-buffer): median {:.3} ms  ({:.2}x vs legacy)",
        opt,
        leg / opt
    );
    println!(
        "Rust f32 optimized:                          median {:.3} ms  ({:.2}x vs legacy)",
        f32t,
        leg / f32t
    );

    if run_python {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let script = std::path::Path::new(manifest_dir)
            .join("../../scripts/benchmark_impute_synthetic.py");
        let script = script.canonicalize().unwrap_or(script);
        let out = Command::new("python3")
            .arg(&script)
            .arg("--n")
            .arg(n.to_string())
            .arg("--p")
            .arg(p.to_string())
            .arg("--knn")
            .arg(knn.to_string())
            .arg("--t")
            .arg(t.to_string())
            .arg("--repeat")
            .arg(repeats.to_string())
            .output();
        match out {
            Ok(o) => {
                if o.status.success() {
                    print!("{}", String::from_utf8_lossy(&o.stdout));
                } else {
                    eprint!("Python stderr: {}", String::from_utf8_lossy(&o.stderr));
                }
            }
            Err(e) => eprintln!("Could not run Python synthetic benchmark: {}", e),
        }
    }
}
