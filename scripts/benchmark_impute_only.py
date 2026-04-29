#!/usr/bin/env python3
"""Time only the diffusion imputation P^t @ X (same op as the Rust SpMM core).

Loads CSR `diff_op` and dense `X` from `rust/magic-impute/fixtures/` (run
`scripts/generate_magic_fixtures.py` first).
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy import sparse as sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture-dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "rust", "magic-impute", "fixtures"
        ),
    )
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    d = os.path.abspath(args.fixture_dir)
    meta = np.load(os.path.join(d, "meta.npy"))
    n, m = int(meta[0]), int(meta[1])
    t = int(meta[4])
    data = np.load(os.path.join(d, "diff_op_data.npy"))
    indices = np.load(os.path.join(d, "diff_op_indices.npy"))
    indptr = np.load(os.path.join(d, "diff_op_indptr.npy"))
    X = np.load(os.path.join(d, "X.npy"))

    P = sp.csr_matrix((data, indices, indptr), shape=(n, m))

    for _ in range(args.warmup):
        Y = X.copy()
        for _ in range(t):
            Y = P.dot(Y)

    times = []
    for _ in range(args.repeat):
        Y = X.copy()
        t0 = time.perf_counter()
        for _ in range(t):
            Y = P.dot(Y)
        times.append(time.perf_counter() - t0)

    arr = np.array(times, dtype=np.float64)
    print(
        "Python scipy.sparse CSR dot dense only (imputation): "
        "mean_s={:.6f} std_s={:.6f} min_s={:.6f} n={} t={} shape={}".format(
            float(arr.mean()),
            float(arr.std()),
            float(arr.min()),
            args.repeat,
            t,
            X.shape,
        )
    )


if __name__ == "__main__":
    main()
