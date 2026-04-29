#!/usr/bin/env python3
"""Regenerate Rust validation fixtures from the reference Python MAGIC pipeline."""

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=os.path.join(REPO_ROOT, "rust", "magic-impute", "fixtures"),
        help="Directory for .npy outputs",
    )
    parser.add_argument(
        "--data",
        default=os.path.join(REPO_ROOT, "data", "test_data.csv"),
        help="Input CSV (same as python/tests)",
    )
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--decay", type=int, default=20)
    parser.add_argument("--t", type=int, default=3)
    parser.add_argument("--n-pca", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    os.chdir(os.path.join(REPO_ROOT, "python"))
    sys.path.insert(0, os.getcwd())

    import magic
    import scprep

    scdata = scprep.io.load_csv(args.data, cell_names=False)
    scdata = scprep.filter.filter_empty_cells(scdata)
    scdata = scprep.filter.filter_empty_genes(scdata)
    scdata = scprep.filter.filter_duplicates(scdata)
    scdata_norm = scprep.normalize.library_size_normalize(scdata)
    scdata_norm = scprep.transform.sqrt(scdata_norm)

    X = scprep.utils.toarray(scdata_norm).astype(np.float64)

    magic_op = magic.MAGIC(
        t=args.t,
        decay=args.decay,
        knn=args.knn,
        n_pca=args.n_pca,
        verbose=False,
        random_state=args.random_state,
        n_jobs=1,
    )
    magic_op.fit(scdata_norm)
    D = magic_op.diff_op
    if sp.issparse(D):
        D = D.tocsr()

    X_magic = magic_op.transform(scdata_norm, genes="all_genes")
    Xm = scprep.utils.toarray(X_magic).astype(np.float64)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "X_magic.npy"), Xm)
    np.save(os.path.join(args.out_dir, "diff_op_data.npy"), D.data.astype(np.float64))
    np.save(os.path.join(args.out_dir, "diff_op_indices.npy"), D.indices.astype(np.int32))
    np.save(os.path.join(args.out_dir, "diff_op_indptr.npy"), D.indptr.astype(np.int32))
    meta = np.array([D.shape[0], D.shape[1], args.knn, args.decay, args.t], dtype=np.int64)
    np.save(os.path.join(args.out_dir, "meta.npy"), meta)
    print(
        "Wrote fixtures to",
        args.out_dir,
        "| X",
        X.shape,
        "| nnz",
        D.nnz,
    )


if __name__ == "__main__":
    main()
