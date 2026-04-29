#!/usr/bin/env python3
"""
Compare Rust `impute_magic` vs scipy CSR.dot dense diffusion on identical synthetic data.

Default cell counts: 1k, 5k, 10k, 100k, 500k (same generators as Rust `magic_impute::synthetic`).

Requires: scipy, numpy. Builds/runs `scale_rust` via cargo unless `--rust-binary` is set.

Example:
  python3 scripts/compare_rust_python_scale.py --repeat 3 --threads 8
  export CARGO_TARGET_DIR=/tmp/magic-target CARGO_INCREMENTAL=0
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent


def synthetic_csr(n: int, knn: int, seed: int):
    indptr = [0]
    indices = []
    data = []
    rng = seed & 0xFFFFFFFFFFFFFFFF
    for _ in range(n):
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
    return sp.csr_matrix((data, indices, indptr), shape=(n, n))


def synthetic_x(n: int, p: int, seed: int = 1):
    x = np.zeros((n, p), dtype=np.float64)
    s = seed & 0xFFFFFFFFFFFFFFFF
    for i in range(n):
        for j in range(p):
            s = (s * 1103515245 + 12345) & 0xFFFFFFFFFFFFFFFF
            x[i, j] = float(s) / float(2**64 - 1) * 10.0
    return x


def median(a: np.ndarray) -> float:
    return float(np.median(a))


def bench_python(n: int, p: int, knn: int, t: int, repeat: int, warmup: int) -> float:
    p_csr = synthetic_csr(n, knn, 42)
    x0 = synthetic_x(n, p, 1)
    for _ in range(warmup):
        y = x0.copy()
        for _ in range(t):
            y = p_csr.dot(y)
    times = []
    for _ in range(repeat):
        y = x0.copy()
        t0 = time.perf_counter()
        for _ in range(t):
            y = p_csr.dot(y)
        times.append(time.perf_counter() - t0)
    return median(np.array(times, dtype=np.float64))


def find_scale_rust_binary(args_binary: str | None, cargo_target_dir: str | None) -> Path:
    if args_binary:
        return Path(args_binary).resolve()
    root = Path(cargo_target_dir) if cargo_target_dir else REPO_ROOT / "rust" / "target"
    return (root / "release" / "scale_rust").resolve()


def run_rust_scale(args: argparse.Namespace, cells: list[int]) -> dict[int, float]:
    exe = find_scale_rust_binary(args.rust_binary, args.cargo_target_dir)
    if not exe.is_file():
        cmd = [
            "cargo",
            "build",
            "--release",
            "-p",
            "magic-impute",
            "--bin",
            "scale_rust",
        ]
        env = os.environ.copy()
        if args.cargo_target_dir:
            env["CARGO_TARGET_DIR"] = args.cargo_target_dir
        if args.cargo_incremental is not None:
            env["CARGO_INCREMENTAL"] = args.cargo_incremental
        print("Building scale_rust:", " ".join(cmd), file=sys.stderr)
        subprocess.run(
            cmd,
            cwd=REPO_ROOT / "rust",
            check=True,
            env=env,
        )
        exe = find_scale_rust_binary(None, args.cargo_target_dir)
        if not exe.is_file():
            raise FileNotFoundError(f"Expected {exe} after cargo build")

    cells_arg = ",".join(str(c) for c in cells)
    cmd = [
        str(exe),
        "--cells",
        cells_arg,
        "--p",
        str(args.p),
        "--knn",
        str(args.knn),
        "--t",
        str(args.t),
        "--threads",
        str(args.threads),
        "--repeat",
        str(args.repeat),
        "--warmup",
        str(args.warmup),
        "--block",
        str(args.block),
    ]
    print("Rust:", " ".join(cmd), file=sys.stderr)
    out = subprocess.check_output(cmd, text=True, env=os.environ)
    result: dict[int, float] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"cells\s+(\d+)\s+median_s\s+([\d.eE+-]+)", line)
        if m:
            result[int(m.group(1))] = float(m.group(2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cells",
        default="1000,5000,10000,100000,500000",
        help="Comma-separated cell counts",
    )
    ap.add_argument("--p", type=int, default=512)
    ap.add_argument("--knn", type=int, default=32)
    ap.add_argument("--t", type=int, default=3)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--skip-rust", action="store_true")
    ap.add_argument("--skip-python", action="store_true")
    ap.add_argument("--rust-binary", default=None, help="Path to scale_rust executable")
    ap.add_argument(
        "--cargo-target-dir",
        default=os.environ.get("CARGO_TARGET_DIR"),
        help="Forwarded when auto-building scale_rust",
    )
    ap.add_argument(
        "--cargo-incremental",
        default=os.environ.get("CARGO_INCREMENTAL", "0"),
    )
    ap.add_argument(
        "--python-max-cells",
        type=int,
        default=None,
        help="Skip Python timing for n above this (memory)",
    )
    args = ap.parse_args()

    cells = [int(x.strip()) for x in args.cells.split(",") if x.strip()]

    rust_s: dict[int, float] = {}
    if not args.skip_rust:
        rust_s = run_rust_scale(args, cells)

    py_max = args.python_max_cells
    print("", file=sys.stderr)
    print(
        "| cells | Rust median_s | Python median_s | Rust vs Python (× faster) |"
    )
    print(
        "|------:|--------------:|----------------:|----------------------------:|"
    )

    for n in cells:
        rs = rust_s.get(n)
        if args.skip_python or (py_max is not None and n > py_max):
            ps = None
        else:
            ps = bench_python(n, args.p, args.knn, args.t, args.repeat, args.warmup)

        if rs is not None and ps is not None and ps > 0:
            ratio = ps / rs
            ratio_s = f"{ratio:.2f}"
        elif rs is not None and ps is not None:
            ratio_s = "—"
        else:
            ratio_s = "—"

        rs_s = f"{rs:.6f}" if rs is not None else "—"
        py_s = f"{ps:.6f}" if ps is not None else "—"
        print(f"| {n} | {rs_s} | {py_s} | {ratio_s} |")

    print("\nParameters: p={} knn={} t={} threads={} repeat={} warmup={}".format(
        args.p, args.knn, args.t, args.threads, args.repeat, args.warmup
    ), file=sys.stderr)


if __name__ == "__main__":
    main()
