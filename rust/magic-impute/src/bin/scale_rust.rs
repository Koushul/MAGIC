//! Emit median seconds for `impute_magic` per cell count (machine-readable for Python driver).

use magic_impute::{impute_magic, synthetic, ImputeConfig};
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn main() {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut cells_list: Option<Vec<usize>> = None;
    let mut args_filtered = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--cells" && i + 1 < args.len() {
            cells_list = Some(
                args[i + 1]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect(),
            );
            i += 2;
            continue;
        }
        args_filtered.push(args[i].clone());
        i += 1;
    }
    args = args_filtered;

    let cells = cells_list.unwrap_or_else(|| vec![1000, 5000, 10_000, 100_000, 500_000]);

    let p = parse_usize(&args, "--p", 512);
    let knn = parse_usize(&args, "--knn", 32);
    let t = parse_u32(&args, "--t", 3);
    let threads = parse_usize(&args, "--threads", num_cpus());
    let repeats = parse_usize(&args, "--repeat", 5);
    let warmup = parse_usize(&args, "--warmup", 1);
    let block = parse_usize(&args, "--block", 128);

    let cfg = ImputeConfig {
        threads: Some(threads),
        gene_block_size: block,
    };

    for &n in &cells {
        let csr = synthetic::csr_row_stochastic(n, knn, 42);
        let x = synthetic::dense_expression(n, p, 1);

        for _ in 0..warmup {
            let _ = impute_magic(&csr, &x, t, &cfg);
        }

        let mut samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let t0 = Instant::now();
            let _ = impute_magic(&csr, &x, t, &cfg);
            samples.push(t0.elapsed().as_secs_f64());
        }
        let med = median(samples);
        println!("cells\t{}\tmedian_s\t{:.9}", n, med);
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(4)
}

fn parse_usize(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|s| s == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_u32(args: &[String], key: &str, default: u32) -> u32 {
    args.iter()
        .position(|s| s == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
