#!/usr/bin/env python3
"""
Benchmark Scanpy (MAGIC Python imputation) vs Rust imputation on pbmc3k_processed.

Steps timed:
  - preprocess + neighbors + leiden
  - per Leiden cluster: full Python fit_transform vs fit (graph) + Rust magic-impute-cli

Outputs docs/figures/pipeline_scanpy_vs_rust_umap.png
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy import sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-png",
        type=Path,
        default=REPO_ROOT / "docs" / "figures" / "pipeline_scanpy_vs_rust_umap.png",
    )
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--magic-t", type=int, default=3)
    args_cli = ap.parse_args()

    import magic
    import scprep

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from rust_process_h5ad import find_magic_cli, fit_magic_python, fit_magic_rust_from_python_graph, prepare_magic_matrix

    adata = sc.datasets.pbmc3k_processed().copy()
    for col in ("cell_type", "leiden"):
        if col in adata.obs.columns:
            del adata.obs[col]

    times_py = {}
    t0 = time.perf_counter()
    x = adata.X
    if sp.issparse(x):
        mx = float(np.asarray(x.data).max()) if x.nnz else 0.0
    else:
        mx = float(np.asarray(x).max())
    if mx > 50:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.leiden(adata, resolution=1.0, key_added="leiden")
    times_py["clustering_s"] = time.perf_counter() - t0

    categories = list(adata.obs["leiden"].astype("category").cat.categories)
    x_den = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)

    magic_py_total = 0.0
    magic_graph_total = 0.0
    magic_rust_impute_total = 0.0

    cli = find_magic_cli()
    if cli is None:
        subprocess.run(
            ["cargo", "build", "--release", "-p", "magic-impute", "--bin", "magic-impute-cli"],
            cwd=REPO_ROOT / "rust",
            check=True,
        )
        cli = find_magic_cli()

    ns = SimpleNamespace(
        magic_knn=10,
        magic_knn_max=None,
        magic_decay=20,
        magic_n_pca=100,
        magic_t=args_cli.magic_t,
        random_state=42,
        magic_n_jobs=1,
        threads=args_cli.threads,
        block=128,
    )

    out_py = np.zeros_like(x_den, dtype=np.float64)
    out_rs = np.zeros_like(x_den, dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root = Path(tmp_root)
        for i, ct in enumerate(categories):
            mask = (adata.obs["leiden"] == ct).to_numpy()
            idx = np.nonzero(mask)[0]
            if idx.size == 0:
                continue
            sub_x = x_den[idx]
            x_ls = prepare_magic_matrix(sub_x)

            t0 = time.perf_counter()
            imp_py = fit_magic_python(
                x_ls,
                knn=ns.magic_knn,
                knn_max=ns.magic_knn_max,
                decay=ns.magic_decay,
                n_pca=ns.magic_n_pca,
                t=ns.magic_t,
                random_state=ns.random_state,
                n_jobs=ns.magic_n_jobs,
            )
            magic_py_total += time.perf_counter() - t0

            op = magic.MAGIC(
                knn=ns.magic_knn,
                knn_max=ns.magic_knn_max,
                decay=ns.magic_decay,
                t=ns.magic_t,
                n_pca=ns.magic_n_pca,
                random_state=ns.random_state,
                n_jobs=ns.magic_n_jobs,
                verbose=False,
            )
            t0 = time.perf_counter()
            op.fit(x_ls)
            magic_graph_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            imp_rs = fit_magic_rust_from_python_graph(
                x_ls,
                knn=ns.magic_knn,
                knn_max=ns.magic_knn_max,
                decay=ns.magic_decay,
                n_pca=ns.magic_n_pca,
                t=ns.magic_t,
                random_state=ns.random_state,
                n_jobs=ns.magic_n_jobs,
                threads_rust=ns.threads,
                block=ns.block,
                tmpdir=tmp_root / f"c{i}",
                cli=cli,
            )
            magic_rust_impute_total += time.perf_counter() - t0

            out_py[idx] = imp_py
            out_rs[idx] = imp_rs

    gi = int(np.argmax(np.asarray(x_den).max(axis=0)))
    gene_py = out_py[:, gi]
    gene_rs = out_rs[:, gi]

    xy = adata.obsm["X_umap"]
    leid_codes = adata.obs["leiden"].astype("category").cat.codes

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=150)
    axes[0, 0].scatter(xy[:, 0], xy[:, 1], c=leid_codes, s=3, cmap="tab20", rasterized=True)
    axes[0, 0].set_title("Leiden")
    axes[0, 1].scatter(xy[:, 0], xy[:, 1], c=leid_codes, s=3, cmap="tab20", rasterized=True)
    axes[0, 1].set_title("Leiden (duplicate)")
    axes[1, 0].scatter(xy[:, 0], xy[:, 1], c=gene_py, s=3, cmap="viridis", rasterized=True)
    axes[1, 0].set_title("MAGIC Python")
    axes[1, 1].scatter(xy[:, 0], xy[:, 1], c=gene_rs, s=3, cmap="viridis", rasterized=True)
    axes[1, 1].set_title("MAGIC Rust SPMM")
    for ax in axes.flat:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
    fig.suptitle(
        "PBMC3k | clustering {:.3f}s | Python MAGIC {:.3f}s | graph {:.3f}s + Rust {:.3f}s".format(
            times_py["clustering_s"],
            magic_py_total,
            magic_graph_total,
            magic_rust_impute_total,
        )
    )
    plt.tight_layout()
    args_cli.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args_cli.out_png)
    plt.close()

    print("scanpy:", sc.__version__)
    print("clustering_s:", times_py["clustering_s"])
    print("magic_python_fit_transform_total_s:", magic_py_total)
    print("magic_python_graph_fit_total_s:", magic_graph_total)
    print("magic_rust_spmm_total_s:", magic_rust_impute_total)
    print("Saved:", args_cli.out_png.resolve())


if __name__ == "__main__":
    main()
