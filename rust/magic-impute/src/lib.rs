use ndarray::{Array2, Axis};
use rayon::prelude::*;
use sprs::CsMat;

pub struct CsrF64 {
    pub data: Vec<f64>,
    pub indices: Vec<i32>,
    pub indptr: Vec<i32>,
    pub nrows: usize,
    pub ncols: usize,
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

    pub fn to_sprs_csr(&self) -> CsMat<f64> {
        let indptr_u: Vec<usize> = self.indptr.iter().map(|&v| v as usize).collect();
        let indices_u: Vec<usize> = self.indices.iter().map(|&v| v as usize).collect();
        CsMat::new(
            (self.nrows, self.ncols),
            indptr_u,
            indices_u,
            self.data.clone(),
        )
    }

    pub fn matmul_dense_rhs_parallel(
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
}

fn sparse_power_csr(mat: &CsMat<f64>, t: u32) -> CsMat<f64> {
    assert_eq!(mat.rows(), mat.cols());
    if t == 0 {
        return CsMat::eye(mat.rows());
    }
    let mut acc = mat.clone();
    for _ in 1..t {
        acc = mat * &acc;
    }
    acc
}

fn csr_dense_matmul(csr: &CsMat<f64>, x: &Array2<f64>) -> Array2<f64> {
    let n = csr.rows();
    let p = x.ncols();
    assert_eq!(csr.cols(), x.nrows());
    let mut out = Array2::<f64>::zeros((n, p));
    for (i, row) in csr.outer_iterator().enumerate() {
        for k in 0..p {
            let mut s = 0.0f64;
            for (j, val) in row.iter() {
                s += *val * x[[j, k]];
            }
            out[[i, k]] = s;
        }
    }
    out
}

pub fn impute_magic_exact(
    diff_op: &CsrF64,
    data: &Array2<f64>,
    t: u32,
    nthreads: Option<usize>,
) -> Array2<f64> {
    let n = diff_op.nrows;
    let p = data.ncols();
    assert_eq!(diff_op.ncols, n);
    assert_eq!(data.nrows(), n);

    if t == 0 {
        return data.clone();
    }

    if (t as usize) > 0 && n < p {
        let d = diff_op.to_sprs_csr();
        let d_t = sparse_power_csr(&d, t);
        return csr_dense_matmul(&d_t, data);
    }

    let mut current = data.clone();
    for _ in 0..t {
        current = diff_op.matmul_dense_rhs_parallel(&current, nthreads);
    }
    current
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
        let out = impute_magic_exact(&csr, &x, t, Some(1));

        let tol = 1e-9;
        assert_eq!(out.dim(), expected.dim());
        for (a, ea) in out.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*a, *ea, epsilon = tol);
        }
    }

    #[test]
    fn classic_matrix_power_matches_numpy_style() {
        let n = 5usize;
        let p = 20usize;
        let mut tr = sprs::TriMat::new((n, n));
        for i in 0..n {
            let j = (i + 1) % n;
            tr.add_triplet(i, j, 0.5_f64);
            tr.add_triplet(i, i, 0.5_f64);
        }
        let d = tr.to_csr();
        let csr = CsrF64::from_parts(
            d.data().to_vec(),
            d.indices().to_vec().into_iter().map(|x| x as i32).collect(),
            d.indptr()
                .raw_storage()
                .to_vec()
                .into_iter()
                .map(|x: usize| x as i32)
                .collect(),
            n,
            n,
        );
        let mut rng = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                rng[[i, j]] = (i + j) as f64 * 0.01 + 0.1;
            }
        }
        let t = 4u32;
        let via = impute_magic_exact(&csr, &rng, t, Some(1));
        let d_t = sparse_power_csr(&csr.to_sprs_csr(), t);
        let ref_mul = csr_dense_matmul(&d_t, &rng);
        for (a, b) in via.iter().zip(ref_mul.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }
}
