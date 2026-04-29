//! Deterministic synthetic row-stochastic CSR graphs and dense expression matrices for benchmarks.

use crate::{Array2, CsrF64};

pub fn csr_row_stochastic(n: usize, knn: usize, seed: u64) -> CsrF64 {
    let mut indptr = vec![0i32; n + 1];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    let mut rng = seed;
    for i in 0..n {
        let row_start = data.len();
        let mut row_sum = 0.0f64;
        for _ in 0..knn {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng as usize) % n;
            let v = ((rng >> 32) as f64 / u32::MAX as f64) * 0.5 + 0.1;
            indices.push(j as i32);
            data.push(v);
            row_sum += v;
        }
        for p in row_start..data.len() {
            data[p] /= row_sum;
        }
        indptr[i + 1] = data.len() as i32;
    }
    CsrF64::from_parts(data, indices, indptr, n, n)
}

pub fn dense_expression(n: usize, p: usize, seed: u64) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, p));
    let mut s = seed;
    for i in 0..n {
        for j in 0..p {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            x[[i, j]] = (s as f64 / u64::MAX as f64) * 10.0;
        }
    }
    x
}
