#!/usr/bin/env python3
"""Time scipy CSR @ dense for the same synthetic P, X, t as `compare_magic_timings` (Rust)."""

import argparse
import time

import numpy as np
from scipy import sparse as sp


def synthetic_csr(n, knn, seed):
    indptr = [0]
    indices = []
    data = []
    rng = seed & 0xFFFFFFFFFFFFFFFF
    for _i in range(n):
        row_start = len(data)
        row_sum = 0.0
        for _ in range(knn):
            rng = (rng * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
            j = rng % n
            v = ((rng >> 32) & 0xFFFFFFFF) / float(0xFFFFFFFF) * 0.5 + 0.1
            indices.append(j)
            data.append(v)
            row_sum += v
        for p in range(row_start, len(data)):
            data[p] /= row_sum
        indptr.append(len(data))
    P = sp.csr_matrix((data, indices, indptr), shape=(n, n))
    return P


def synthetic_x(n, p, seed=1):
    x = np.zeros((n, p), dtype=np.float64)
    s = seed & 0xFFFFFFFFFFFFFFFF
    for i in range(n):
        for j in range(p):
            s = (s * 1103515245 + 12345) & 0xFFFFFFFFFFFFFFFF
            x[i, j] = float(s) / float(2**64 - 1) * 10.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--p", type=int, default=512)
    ap.add_argument("--knn", type=int, default=32)
    ap.add_argument("--t", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    P = synthetic_csr(args.n, args.knn, args.seed)
    X = synthetic_x(args.n, args.p, 1)

    for _ in range(args.warmup):
        Y = X.copy()
        for _ in range(args.t):
            Y = P.dot(Y)

    times = []
    for _ in range(args.repeat):
        Y = X.copy()
        t0 = time.perf_counter()
        for _ in range(args.t):
            Y = P.dot(Y)
        times.append(time.perf_counter() - t0)

    arr = np.array(times, dtype=np.float64)
    print(
        "Python scipy CSR.dot dense (synthetic, matches Rust compare_magic_timings): "
        "median_s={:.6f} mean_s={:.6f} n={} shape={} t={}".format(
            float(np.median(arr)),
            float(arr.mean()),
            args.repeat,
            X.shape,
            args.t,
        )
    )


if __name__ == "__main__":
    main()
