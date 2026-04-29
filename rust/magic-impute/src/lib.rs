use ndarray::{s, Array2, Axis};
use rayon::prelude::*;

pub mod synthetic;

pub struct CsrF64 {
    pub data: Vec<f64>,
    pub indices: Vec<i32>,
    pub indptr: Vec<i32>,
    pub nrows: usize,
    pub ncols: usize,
}

#[derive(Clone, Debug)]
pub struct ImputeConfig {
    pub threads: Option<usize>,
    pub gene_block_size: usize,
}

impl Default for ImputeConfig {
    fn default() -> Self {
        Self {
            threads: None,
            gene_block_size: 128,
        }
    }
}

impl CsrF64 {
    pub fn from_parts(
        data: Vec<f64>,
        indices: Vec<i32>,
        indptr: Vec<i32>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        assert_eq!(indptr.len(), nrows + 1);
        Self {
            data,
            indices,
            indptr,
            nrows,
            ncols,
        }
    }

    pub fn matmul_dense_rhs_legacy(
        &self,
        x: &Array2<f64>,
        nthreads: Option<usize>,
    ) -> Array2<f64> {
        assert_eq!(x.nrows(), self.ncols, "CSR cols must match X rows");
        let out_nrows = self.nrows;
        let out_ncols = x.ncols();
        let mut out = Array2::<f64>::zeros((out_nrows, out_ncols));

        let pool = nthreads
            .map(|n| rayon::ThreadPoolBuilder::new().num_threads(n).build().ok())
            .flatten();

        let row_fn = |i: usize, mut out_row: ndarray::ArrayViewMut1<f64>| {
            let r0 = self.indptr[i] as usize;
            let r1 = self.indptr[i + 1] as usize;
            for k in 0..out_ncols {
                let mut acc = 0.0f64;
                for p in r0..r1 {
                    let j = self.indices[p] as usize;
                    acc += self.data[p] * x[[j, k]];
                }
                out_row[k] = acc;
            }
        };

        if let Some(pool) = pool {
            pool.install(|| {
                out.axis_iter_mut(Axis(0))
                    .into_iter()
                    .enumerate()
                    .par_bridge()
                    .for_each(|(i, row)| row_fn(i, row));
            });
        } else {
            out.axis_iter_mut(Axis(0))
                .into_iter()
                .enumerate()
                .par_bridge()
                .for_each(|(i, row)| row_fn(i, row));
        }

        out
    }

    fn spmm_dense_blocked_into(
        &self,
        x: &Array2<f64>,
        out: &mut Array2<f64>,
        config: &ImputeConfig,
    ) {
        assert_eq!(x.nrows(), self.ncols);
        assert_eq!(out.dim(), (self.nrows, x.ncols()));

        let p = x.ncols();
        let bs = config.gene_block_size.max(1);

        for kc in (0..p).step_by(bs) {
            let ke = (kc + bs).min(p);
            let mut out_blk = out.slice_mut(s![.., kc..ke]);
            let x_blk = x.slice(s![.., kc..ke]);
            self.spmm_dense_block_f64(x_blk, &mut out_blk, config);
        }
    }

    fn spmm_dense_block_f64(
        &self,
        x_blk: ndarray::ArrayView2<f64>,
        out_blk: &mut ndarray::ArrayViewMut2<f64>,
        config: &ImputeConfig,
    ) {
        let pool = config
            .threads
            .map(|n| rayon::ThreadPoolBuilder::new().num_threads(n).build().ok())
            .flatten();

        let row_fn = |i: usize, mut out_row: ndarray::ArrayViewMut1<f64>| {
            let r0 = self.indptr[i] as usize;
            let r1 = self.indptr[i + 1] as usize;
            out_row.fill(0.0);
            for ptr in r0..r1 {
                let j = self.indices[ptr] as usize;
                let w = self.data[ptr];
                let x_row = x_blk.row(j);
                for (oc, xc) in out_row.iter_mut().zip(x_row.iter()) {
                    *oc += w * *xc;
                }
            }
        };

        if let Some(ref pool) = pool {
            pool.install(|| {
                out_blk
                    .axis_iter_mut(Axis(0))
                    .into_iter()
                    .enumerate()
                    .par_bridge()
                    .for_each(|(i, row)| row_fn(i, row));
            });
        } else {
            out_blk
                .axis_iter_mut(Axis(0))
                .into_iter()
                .enumerate()
                .par_bridge()
                .for_each(|(i, row)| row_fn(i, row));
        }
    }

    fn spmm_dense_blocked_into_f32(
        &self,
        x: &Array2<f32>,
        out: &mut Array2<f32>,
        config: &ImputeConfig,
    ) {
        assert_eq!(x.nrows(), self.ncols);
        assert_eq!(out.dim(), (self.nrows, x.ncols()));

        let p = x.ncols();
        let bs = config.gene_block_size.max(1);

        for kc in (0..p).step_by(bs) {
            let ke = (kc + bs).min(p);
            let mut out_blk = out.slice_mut(s![.., kc..ke]);
            let x_blk = x.slice(s![.., kc..ke]);
            self.spmm_dense_block_f32(x_blk, &mut out_blk, config);
        }
    }

    fn spmm_dense_block_f32(
        &self,
        x_blk: ndarray::ArrayView2<f32>,
        out_blk: &mut ndarray::ArrayViewMut2<f32>,
        config: &ImputeConfig,
    ) {
        let pool = config
            .threads
            .map(|n| rayon::ThreadPoolBuilder::new().num_threads(n).build().ok())
            .flatten();

        let row_fn = |i: usize, mut out_row: ndarray::ArrayViewMut1<f32>| {
            let r0 = self.indptr[i] as usize;
            let r1 = self.indptr[i + 1] as usize;
            out_row.fill(0.0);
            for ptr in r0..r1 {
                let j = self.indices[ptr] as usize;
                let w = self.data[ptr] as f32;
                let x_row = x_blk.row(j);
                for (oc, xc) in out_row.iter_mut().zip(x_row.iter()) {
                    *oc += w * *xc;
                }
            }
        };

        if let Some(ref pool) = pool {
            pool.install(|| {
                out_blk
                    .axis_iter_mut(Axis(0))
                    .into_iter()
                    .enumerate()
                    .par_bridge()
                    .for_each(|(i, row)| row_fn(i, row));
            });
        } else {
            out_blk
                .axis_iter_mut(Axis(0))
                .into_iter()
                .enumerate()
                .par_bridge()
                .for_each(|(i, row)| row_fn(i, row));
        }
    }
}

pub fn impute_magic(diff_op: &CsrF64, data: &Array2<f64>, t: u32, config: &ImputeConfig) -> Array2<f64> {
    let n = diff_op.nrows;
    let p = data.ncols();
    assert_eq!(diff_op.ncols, n);
    assert_eq!(data.nrows(), n);

    if t == 0 {
        return data.clone();
    }

    let mut a = data.clone();
    let mut b = Array2::<f64>::zeros((n, p));
    let mut from_a = true;

    for _ in 0..t {
        if from_a {
            diff_op.spmm_dense_blocked_into(&a, &mut b, config);
        } else {
            diff_op.spmm_dense_blocked_into(&b, &mut a, config);
        }
        from_a = !from_a;
    }

    if from_a {
        a
    } else {
        b
    }
}

pub fn impute_magic_f32(diff_op: &CsrF64, data: &Array2<f32>, t: u32, config: &ImputeConfig) -> Array2<f32> {
    let n = diff_op.nrows;
    let p = data.ncols();
    assert_eq!(diff_op.ncols, n);
    assert_eq!(data.nrows(), n);

    if t == 0 {
        return data.clone();
    }

    let mut a = data.clone();
    let mut b = Array2::<f32>::zeros((n, p));
    let mut from_a = true;

    for _ in 0..t {
        if from_a {
            diff_op.spmm_dense_blocked_into_f32(&a, &mut b, config);
        } else {
            diff_op.spmm_dense_blocked_into_f32(&b, &mut a, config);
        }
        from_a = !from_a;
    }

    if from_a {
        a
    } else {
        b
    }
}

pub fn impute_magic_legacy(diff_op: &CsrF64, data: &Array2<f64>, t: u32, nthreads: Option<usize>) -> Array2<f64> {
    let n = diff_op.nrows;
    assert_eq!(diff_op.ncols, n);
    assert_eq!(data.nrows(), n);

    if t == 0 {
        return data.clone();
    }

    let mut current = data.clone();
    for _ in 0..t {
        current = diff_op.matmul_dense_rhs_legacy(&current, nthreads);
    }
    current
}

pub fn impute_magic_exact(
    diff_op: &CsrF64,
    data: &Array2<f64>,
    t: u32,
    nthreads: Option<usize>,
) -> Array2<f64> {
    let mut cfg = ImputeConfig::default();
    cfg.threads = nthreads;
    impute_magic(diff_op, data, t, &cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;
    use ndarray_npy::read_npy;
    use std::path::PathBuf;

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
    }

    #[test]
    fn matches_python_reference_fixture() {
        let dir = fixture_dir();
        let x: Array2<f64> = read_npy(dir.join("X.npy")).expect("X.npy");
        let expected: Array2<f64> = read_npy(dir.join("X_magic.npy")).expect("X_magic.npy");
        let meta: ndarray::Array1<i64> = read_npy(dir.join("meta.npy")).expect("meta");
        let n = meta[0] as usize;
        let m = meta[1] as usize;
        let t = meta[4] as u32;

        let data_arr: ndarray::Array1<f64> = read_npy(dir.join("diff_op_data.npy")).expect("parse data");
        let ind_arr: ndarray::Array1<i32> = read_npy(dir.join("diff_op_indices.npy")).expect("parse ind");
        let ptr_arr: ndarray::Array1<i32> = read_npy(dir.join("diff_op_indptr.npy")).expect("parse ptr");

        let data_f = data_arr.into_raw_vec_and_offset().0;
        let indices = ind_arr.into_raw_vec_and_offset().0;
        let indptr = ptr_arr.into_raw_vec_and_offset().0;

        let csr = CsrF64::from_parts(data_f, indices, indptr, n, m);
        let cfg = ImputeConfig {
            threads: Some(1),
            gene_block_size: 64,
        };
        let out = impute_magic(&csr, &x, t, &cfg);

        let tol = 1e-9;
        assert_eq!(out.dim(), expected.dim());
        for (a, ea) in out.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*a, *ea, epsilon = tol);
        }
    }

    #[test]
    fn blocked_matches_legacy_random() {
        let n = 64usize;
        let p = 97usize;
        let csr = crate::synthetic::csr_row_stochastic(n, 8, 99);
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                x[[i, j]] = ((i * 31 + j * 7) % 100) as f64 * 0.01;
            }
        }
        let t = 4u32;
        let leg = impute_magic_legacy(&csr, &x, t, Some(1));
        let cfg = ImputeConfig {
            threads: Some(1),
            gene_block_size: 17,
        };
        let opt = impute_magic(&csr, &x, t, &cfg);
        for (a, b) in leg.iter().zip(opt.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn f32_near_f64_fixture() {
        let dir = fixture_dir();
        let x64: Array2<f64> = read_npy(dir.join("X.npy")).expect("X.npy");
        let expected: Array2<f64> = read_npy(dir.join("X_magic.npy")).expect("X_magic.npy");
        let meta: ndarray::Array1<i64> = read_npy(dir.join("meta.npy")).expect("meta");
        let n = meta[0] as usize;
        let m = meta[1] as usize;
        let t = meta[4] as u32;

        let data_arr: ndarray::Array1<f64> = read_npy(dir.join("diff_op_data.npy")).expect("parse data");
        let ind_arr: ndarray::Array1<i32> = read_npy(dir.join("diff_op_indices.npy")).expect("parse ind");
        let ptr_arr: ndarray::Array1<i32> = read_npy(dir.join("diff_op_indptr.npy")).expect("parse ptr");

        let data_f = data_arr.into_raw_vec_and_offset().0;
        let indices = ind_arr.into_raw_vec_and_offset().0;
        let indptr = ptr_arr.into_raw_vec_and_offset().0;

        let csr = CsrF64::from_parts(data_f, indices, indptr, n, m);
        let x32 = x64.mapv(|v| v as f32);
        let cfg = ImputeConfig {
            threads: Some(1),
            gene_block_size: 64,
        };
        let out32 = impute_magic_f32(&csr, &x32, t, &cfg);
        let rtol = 1e-5f64;
        let atol = 1e-6f64;
        for (o, e) in out32.iter().zip(expected.iter()) {
            assert!((*o as f64 - *e).abs() <= atol + rtol * e.abs());
        }
    }
}
