#!/usr/bin/env python3
"""
Benchmark the h5ad pipeline in Scanpy/Python mode versus Rust-assisted mode.

Both runs use ``scripts/rust_process_h5ad.py`` on the same h5ad input with
``obs['cell_type']``, ``obs['leiden']``, and ``layers['imputed_count']`` removed,
forcing the full preprocess → neighbors → Leiden proxy → per-cluster MAGIC path.

Outputs docs/figures/pipeline_scanpy_vs_rust_umap.png
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parent.parent


def pipeline_args(args_cli, *, use_rust_leiden: bool) -> SimpleNamespace:
    return SimpleNamespace(
        neighbors=args_cli.neighbors,
        leiden_resolution=args_cli.leiden_resolution,
        rust_leiden_resolution=args_cli.rust_leiden_resolution,
        leiden_randomness=args_cli.leiden_randomness,
        leiden_max_iter=args_cli.leiden_max_iter,
        hvg=args_cli.hvg,
        n_top_hvg=args_cli.n_top_hvg,
        magic_knn=args_cli.magic_knn,
        magic_knn_max=args_cli.magic_knn_max,
        magic_decay=args_cli.magic_decay,
        magic_n_pca=args_cli.magic_n_pca,
        magic_t=args_cli.magic_t,
        random_state=args_cli.random_state,
        magic_n_jobs=args_cli.magic_n_jobs,
        threads=args_cli.threads,
        cluster_workers=args_cli.cluster_workers,
        block=args_cli.block,
        use_rust_leiden=use_rust_leiden,
        python_leiden_only=not use_rust_leiden,
    )


def clean_input_adata():
    adata = sc.datasets.pbmc3k_processed().copy()
    for col in ("cell_type", "leiden"):
        if col in adata.obs.columns:
            del adata.obs[col]
    for layer in ("imputed", "imputed_count", "imputation_count"):
        if layer in adata.layers:
            del adata.layers[layer]
    return adata


def timed_pipeline(input_h5ad: Path, output_h5ad: Path, ns, *, use_rust_magic: bool):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from rust_process_h5ad import process_h5ad

    benchmark = {}
    t0 = time.perf_counter()
    process_h5ad(input_h5ad, output_h5ad, ns, use_rust_magic=use_rust_magic, benchmark=benchmark)
    benchmark["total_s"] = time.perf_counter() - t0
    return benchmark, sc.read_h5ad(output_h5ad)


def print_timings(name: str, timings: dict) -> None:
    print(f"{name}:")
    for key in (
        "cell_type_source",
        "n_leiden_clusters",
        "leiden_preprocess_s",
        "neighbors_s",
        "leiden_scanpy_s",
        "leiden_rust_s",
        "ensure_cell_type_s",
        "magic_total_s",
        "total_s",
    ):
        if key in timings:
            print(f"  {key}: {timings[key]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-png",
        type=Path,
        default=REPO_ROOT / "docs" / "figures" / "pipeline_scanpy_vs_rust_umap.png",
    )
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument("--leiden-resolution", type=float, default=1.0)
    ap.add_argument("--rust-leiden-resolution", type=float, default=None)
    ap.add_argument("--leiden-randomness", type=float, default=0.01)
    ap.add_argument("--leiden-max-iter", type=int, default=100)
    ap.add_argument("--hvg", action="store_true")
    ap.add_argument("--n-top-hvg", type=int, default=2000)
    ap.add_argument("--magic-knn", type=int, default=10)
    ap.add_argument("--magic-knn-max", type=int, default=None)
    ap.add_argument("--magic-decay", type=int, default=20)
    ap.add_argument("--magic-n-pca", type=int, default=100)
    ap.add_argument("--magic-t", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--magic-n-jobs", type=int, default=1)
    ap.add_argument("--cluster-workers", type=int, default=None)
    ap.add_argument("--block", type=int, default=128)
    args_cli = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root = Path(tmp_root)
        input_h5ad = tmp_root / "input.h5ad"
        clean_input_adata().write_h5ad(input_h5ad)
        py_ns = pipeline_args(args_cli, use_rust_leiden=False)
        rs_ns = pipeline_args(args_cli, use_rust_leiden=True)
        timings_py, adata_py = timed_pipeline(
            input_h5ad,
            tmp_root / "python_pipeline.h5ad",
            py_ns,
            use_rust_magic=False,
        )
        timings_rs, adata_rs = timed_pipeline(
            input_h5ad,
            tmp_root / "rust_pipeline.h5ad",
            rs_ns,
            use_rust_magic=True,
        )

    timings_py["magic_total_s"] = float(sum(timings_py.get("magic_cluster_times", [])))
    timings_rs["magic_total_s"] = float(sum(timings_rs.get("magic_cluster_times", [])))

    xy = adata_py.obsm["X_umap"]
    py_leiden = adata_py.obs["leiden"].astype("category").cat.codes.to_numpy()
    rs_leiden = adata_rs.obs["leiden"].astype("category").cat.codes.to_numpy()
    gi = int(np.argmax(np.asarray(adata_py.X).max(axis=0)))
    gene_py = np.asarray(adata_py.layers["imputed"])[:, gi]
    gene_rs = np.asarray(adata_rs.layers["imputed"])[:, gi]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=150)
    axes[0, 0].scatter(xy[:, 0], xy[:, 1], c=py_leiden, s=3, cmap="tab20", rasterized=True)
    axes[0, 0].set_title("Scanpy Leiden")
    axes[0, 1].scatter(xy[:, 0], xy[:, 1], c=rs_leiden, s=3, cmap="tab20", rasterized=True)
    axes[0, 1].set_title("Rust Leiden")
    axes[1, 0].scatter(xy[:, 0], xy[:, 1], c=gene_py, s=3, cmap="viridis", rasterized=True)
    axes[1, 0].set_title("Python MAGIC")
    axes[1, 1].scatter(xy[:, 0], xy[:, 1], c=gene_rs, s=3, cmap="viridis", rasterized=True)
    axes[1, 1].set_title("Rust MAGIC SPMM")
    for ax in axes.flat:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
    fig.suptitle(
        "PBMC3k | Scanpy {} | Python total {:.3f}s vs Rust total {:.3f}s".format(
            sc.__version__,
            timings_py["total_s"],
            timings_rs["total_s"],
        )
    )
    plt.tight_layout()
    args_cli.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args_cli.out_png)
    plt.close()

    print("scanpy_version:", sc.__version__)
    print_timings("python_pipeline", timings_py)
    print_timings("rust_pipeline", timings_rs)
    print("Saved:", args_cli.out_png.resolve())


if __name__ == "__main__":
    main()
