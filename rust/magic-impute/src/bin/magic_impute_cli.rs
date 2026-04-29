use magic_impute::{impute_magic, CsrF64, ImputeConfig};
use ndarray::Array2;
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut threads = None;
    let mut block = 128usize;
    let mut t = 3u32;
    args.retain(|a| {
        if let Some(rest) = a.strip_prefix("--threads=") {
            threads = rest.parse().ok();
            return false;
        }
        if let Some(rest) = a.strip_prefix("--block=") {
            block = rest.parse().unwrap_or(128);
            return false;
        }
        if let Some(rest) = a.strip_prefix("--t=") {
            t = rest.parse().unwrap_or(3);
            return false;
        }
        true
    });

    let mut indptr_p: Option<&Path> = None;
    let mut indices_p: Option<&Path> = None;
    let mut data_p: Option<&Path> = None;
    let mut n = 0usize;
    let mut m = 0usize;
    let mut x_p: Option<&Path> = None;
    let mut out_p: Option<&Path> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--indptr" => {
                indptr_p = Some(Path::new(&args[i + 1]));
                i += 2;
            }
            "--indices" => {
                indices_p = Some(Path::new(&args[i + 1]));
                i += 2;
            }
            "--data" => {
                data_p = Some(Path::new(&args[i + 1]));
                i += 2;
            }
            "--shape" => {
                n = args[i + 1].parse().unwrap_or(0);
                m = args[i + 2].parse().unwrap_or(0);
                i += 3;
            }
            "--x" => {
                x_p = Some(Path::new(&args[i + 1]));
                i += 2;
            }
            "--out" => {
                out_p = Some(Path::new(&args[i + 1]));
                i += 2;
            }
            _ => {
                eprintln!("Unknown arg {}", args[i]);
                std::process::exit(2);
            }
        }
    }

    let indptr_p = indptr_p.expect("--indptr");
    let indices_p = indices_p.expect("--indices");
    let data_p = data_p.expect("--data");
    let x_p = x_p.expect("--x");
    let out_p = out_p.expect("--out");

    let ip: ndarray::Array1<i64> = ndarray_npy::read_npy(indptr_p).expect("indptr");
    let ij: ndarray::Array1<i32> = ndarray_npy::read_npy(indices_p).expect("indices");
    let csr_values: ndarray::Array1<f64> = ndarray_npy::read_npy(data_p).expect("data");
    let x: Array2<f64> = ndarray_npy::read_npy(x_p).expect("x");

    let indptr_i32: Vec<i32> = ip.iter().map(|&v| v as i32).collect();
    let indices_vec = ij.into_raw_vec_and_offset().0;
    let data_vec = csr_values.into_raw_vec_and_offset().0;
    let csr = CsrF64::from_parts(data_vec, indices_vec, indptr_i32, n, m);

    let cfg = ImputeConfig {
        threads,
        gene_block_size: block,
    };
    let out = impute_magic(&csr, &x, t, &cfg);
    ndarray_npy::write_npy(out_p, &out).expect("write out");
}
