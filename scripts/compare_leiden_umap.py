#!/usr/bin/env python3
"""
Compare Scanpy Leiden vs vendored scan-rs Leiden (Rust) on the same neighbor graph.

Scanpy uses ``leidenalg`` (RBConfiguration) by default; Rust uses CPM with a
different resolution scale. This script **binary-searches** the Rust resolution
so both runs yield the **same number of clusters**, then reports ARI/NMI and timing.

Exports symmetric connectivities as upper-triangle CSR for Rust (undirected edges once).

Requires: scanpy, numpy, scipy, scikit-learn, matplotlib

Example:
  export CARGO_TARGET_DIR=/tmp/magic-cargo-target CARGO_INCREMENTAL=0
  python3 scripts/compare_leiden_umap.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy import sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO_ROOT = Path(__file__).resolve().parent.parent


def csr_upper_triangle(conn: sp.csr_matrix) -> sp.csr_matrix:
    conn = conn.tocsr()
    coo = conn.tocoo()
    rows, cols, data = [], [], []
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if i < j and v > 0:
            rows.append(i)
            cols.append(j)
            data.append(float(v))
    if not rows:
        return conn
    return sp.coo_matrix((data, (rows, cols)), shape=conn.shape).tocsr()


def run_rust_leiden(
    exe: Path,
    indptr: Path,
    indices: Path,
    data: Path,
    n_nodes: int,
    resolution: float,
    randomness: float,
    seed: int,
    max_iter: int,
    labels_out: Path | None,
    *,
    probe_count_only: bool = False,
) -> None:
    cmd = [
        str(exe),
        "--indptr",
        str(indptr),
        "--indices",
        str(indices),
        "--data",
        str(data),
        "--shape",
        str(n_nodes),
        "--resolution",
        str(resolution),
        "--randomness",
        str(randomness),
        "--seed",
        str(seed),
        "--max-iter",
        str(max_iter),
    ]
    if probe_count_only:
        cmd.append("--probe-count-only")
    else:
        cmd.extend(["--out-labels", str(labels_out)])
    subprocess.run(cmd, check=True, capture_output=True)


def rust_cluster_count(
    exe: Path,
    td: Path,
    n_nodes: int,
    resolution: float,
    randomness: float,
    seed: int,
    max_iter: int,
) -> int:
    out = subprocess.check_output(
        [
            str(exe),
            "--indptr",
            str(td / "indptr.npy"),
            "--indices",
            str(td / "indices.npy"),
            "--data",
            str(td / "data.npy"),
            "--shape",
            str(n_nodes),
            "--resolution",
            str(resolution),
            "--randomness",
            str(randomness),
            "--seed",
            str(seed),
            "--max-iter",
            str(max_iter),
            "--probe-count-only",
        ],
        text=True,
    )
    return int(out.strip())


def count_clusters_from_rust(
    exe: Path,
    td: Path,
    n_nodes: int,
    resolution: float,
    randomness: float,
    seed: int,
    max_iter: int,
) -> int:
    return rust_cluster_count(exe, td, n_nodes, resolution, randomness, seed, max_iter)


def binary_search_rust_resolution(
    exe: Path,
    td: Path,
    n_nodes: int,
    k_target: int,
    randomness: float,
    seed: int,
    max_iter: int,
) -> float:
    lo, hi = 1e-14, 50.0
    best_r = 1e-6
    best_diff = 10**9
    for _ in range(52):
        mid = 0.5 * (lo + hi)
        k = count_clusters_from_rust(exe, td, n_nodes, mid, randomness, seed, max_iter)
        diff = abs(k - k_target)
        if diff < best_diff:
            best_diff = diff
            best_r = mid
        if k == k_target:
            return mid
        if k > k_target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-18:
            break
    return best_r


def find_leiden_exe() -> Path:
    env_bin = os.environ.get("LEIDEN_FROM_CSR")
    if env_bin:
        return Path(env_bin).resolve()
    ctd = os.environ.get("CARGO_TARGET_DIR", "")
    if ctd:
        p = Path(ctd) / "release" / "leiden-from-csr"
        if p.is_file():
            return p
    p = REPO_ROOT / "rust" / "target" / "release" / "leiden-from-csr"
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=float, default=1.0, help="Scanpy Leiden resolution")
    ap.add_argument(
        "--out-png",
        type=Path,
        default=REPO_ROOT / "docs" / "figures" / "leiden_scanpy_vs_rust.png",
    )
    ap.add_argument("--randomness", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument(
        "--rust-resolution",
        type=float,
        default=None,
        help="If set, skip binary search and use this CPM resolution in Rust",
    )
    ap.add_argument(
        "--flavor",
        choices=("leidenalg", "igraph"),
        default="leidenalg",
        help="Scanpy Leiden backend",
    )
    ap.add_argument(
        "--rust-timing-repeats",
        type=int,
        default=7,
        help="Median Rust timing uses this many runs of the final clustering (after resolution search)",
    )
    ap.add_argument(
        "--scanpy-timing-repeats",
        type=int,
        default=7,
        help="Median Scanpy timing uses this many runs",
    )
    args = ap.parse_args()

    sc.settings.verbosity = 1

    adata = sc.datasets.pbmc3k_processed()
    if "connectivities" not in adata.obsp:
        sc.pp.neighbors(adata, n_neighbors=args.neighbors)

    conn = adata.obsp["connectivities"].tocsr()
    g = csr_upper_triangle(conn)
    n_nodes = adata.n_obs

    scanpy_times = []
    for _ in range(max(1, args.scanpy_timing_repeats)):
        t0 = time.perf_counter()
        kw = dict(resolution=args.resolution, key_added="leiden_scanpy", flavor=args.flavor)
        if args.flavor == "igraph":
            kw["n_iterations"] = 2
        sc.tl.leiden(adata, **kw)
        scanpy_times.append(time.perf_counter() - t0)
    t_scanpy = float(np.median(np.array(scanpy_times, dtype=np.float64)))

    k_target = int(adata.obs["leiden_scanpy"].astype("category").cat.categories.size)

    indptr = np.asarray(g.indptr, dtype=np.int64)
    indices = np.asarray(g.indices, dtype=np.int32)
    data = np.asarray(g.data, dtype=np.float64)

    exe = find_leiden_exe()
    if not exe.is_file():
        build = [
            "cargo",
            "build",
            "--release",
            "-p",
            "leiden-tools",
            "--bin",
            "leiden-from-csr",
        ]
        env = os.environ.copy()
        subprocess.run(build, cwd=REPO_ROOT / "rust", check=True, env=env)
        exe = find_leiden_exe()
    if not exe.is_file():
        raise FileNotFoundError(f"Build leiden-from-csr first; expected {exe}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        np.save(td / "indptr.npy", indptr)
        np.save(td / "indices.npy", indices)
        np.save(td / "data.npy", data)

        if args.rust_resolution is not None:
            r_rust = float(args.rust_resolution)
        else:
            r_rust = binary_search_rust_resolution(
                exe,
                td,
                n_nodes,
                k_target,
                args.randomness,
                args.seed,
                args.max_iter,
            )

        lab_rust = td / "labels_rust_final.npy"
        rust_times = []
        n_rep = max(1, args.rust_timing_repeats)
        for rep in range(n_rep):
            t0 = time.perf_counter()
            run_rust_leiden(
                exe,
                td / "indptr.npy",
                td / "indices.npy",
                td / "data.npy",
                n_nodes,
                r_rust,
                args.randomness,
                args.seed,
                args.max_iter,
                lab_rust,
                probe_count_only=False,
            )
            rust_times.append(time.perf_counter() - t0)
        t_rust = float(np.median(np.array(rust_times, dtype=np.float64)))
        labels_rust = np.load(lab_rust)

    leid = adata.obs["leiden_scanpy"].astype("category")
    y_py = leid.cat.codes.to_numpy()
    y_rs = labels_rust.astype(np.int64)

    ari = adjusted_rand_score(y_py, y_rs)
    nmi = normalized_mutual_info_score(y_py, y_rs)
    k_rust = len(np.unique(y_rs))

    if "X_umap" in adata.obsm:
        xy = adata.obsm["X_umap"]
    else:
        sc.tl.umap(adata)
        xy = adata.obsm["X_umap"]

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    sc_plot = leid.cat.codes
    axes[0].scatter(xy[:, 0], xy[:, 1], c=sc_plot, s=4, cmap="tab20", rasterized=True)
    axes[0].set_title("Scanpy Leiden ({})".format(args.flavor))
    axes[1].scatter(xy[:, 0], xy[:, 1], c=y_rs, s=4, cmap="tab20", rasterized=True)
    axes[1].set_title("Rust Leiden (scan-rs CPM, r={:.4g})".format(r_rust))
    for ax in axes:
        ax.set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")
    fig.suptitle(
        "PBMC3k processed | k_scanpy={} k_rust={} | ARI={:.3f} NMI={:.3f} | "
        "median time: Scanpy {:.4f}s vs Rust {:.4f}s".format(
            k_target, k_rust, ari, nmi, t_scanpy, t_rust
        )
    )
    plt.tight_layout()
    fig.savefig(args.out_png)
    plt.close()

    print("compare_leiden_umap: n_cells={} n_edges_upper={} scanpy_res={}".format(
        n_nodes, g.nnz, args.resolution
    ))
    print("  Rust CPM resolution used: {:.8g} (Scanpy res stays {:.4g})".format(r_rust, args.resolution))
    print("  Scanpy Leiden median wall time: {:.6f} s ({} runs)".format(
        t_scanpy, max(1, args.scanpy_timing_repeats)
    ))
    print("  Rust Leiden median wall time:   {:.6f} s ({} runs)".format(
        t_rust, max(1, args.rust_timing_repeats)
    ))
    print("  Clusters: Scanpy {} | Rust {}".format(k_target, k_rust))
    print("  Adjusted Rand Index:     {:.6f}".format(ari))
    print("  NMI:                     {:.6f}".format(nmi))
    print("  Saved:", args.out_png.resolve())


if __name__ == "__main__":
    main()
