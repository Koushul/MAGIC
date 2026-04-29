#!/usr/bin/env python3
"""
AnnData pipeline: optional QC → cell labels (cell_type or Leiden proxy) → per-cluster MAGIC.

**Trigger:** run full pipeline only when ``layers['imputed_count']`` is **missing**.
If present, write ``output`` as copy/skip (no work).

**MAGIC skip:** if ``layers['imputation_count']`` exists, clustering still runs if needed,
but **MAGIC imputation is skipped** (only placeholder ``imputed_count``).

Leiden / ``cell_type``:
  - If ``obs['cell_type']`` exists → use for per-cluster MAGIC.
  - Else if ``obs['leiden']`` exists → copy to ``obs['cell_type']``.
  - Else → neighbors + Rust Leiden on **log-normalized** data (skips normalize+log if X looks already log).

MAGIC writes ``layers['imputed']`` and ``layers['imputed_count']`` (cluster index per cell).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse as sp

try:
    import magic
    import scprep
except ImportError as e:
    print("Install: pip install magic-impute scprep anndata", file=sys.stderr)
    raise e


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUST_LEIDEN_RESOLUTION_SCALE = 0.0019073486


def find_magic_cli() -> Path | None:
    env = os.environ.get("MAGIC_IMPUTE_CLI")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    ctd = os.environ.get("CARGO_TARGET_DIR", "")
    if ctd:
        p = Path(ctd) / "release" / "magic-impute-cli"
        if p.is_file():
            return p
    p = REPO_ROOT / "rust" / "target" / "release" / "magic-impute-cli"
    return p if p.is_file() else None


def find_leiden_cli() -> Path | None:
    env = os.environ.get("LEIDEN_FROM_CSR")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    ctd = os.environ.get("CARGO_TARGET_DIR", "")
    if ctd:
        p = Path(ctd) / "release" / "leiden-from-csr"
        if p.is_file():
            return p
    p = REPO_ROOT / "rust" / "target" / "release" / "leiden-from-csr"
    return p if p.is_file() else None


def prepare_magic_matrix(sub_x: np.ndarray) -> np.ndarray:
    """sqrt-library normalize for counts; clip non-negative for scaled/log data."""
    sub_x = np.maximum(np.asarray(sub_x, dtype=np.float64), 0.0)
    x_ls = scprep.normalize.library_size_normalize(sub_x)
    return scprep.transform.sqrt(x_ls)


def dense_float64_row_major(x):
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64, order="C")


def csr_upper_triangle(conn: sp.csr_matrix) -> sp.csr_matrix:
    conn = conn.tocsr()
    coo = conn.tocoo()
    keep = (coo.row < coo.col) & (coo.data > 0)
    if not np.any(keep):
        return conn
    return sp.coo_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])),
        shape=conn.shape,
    ).tocsr()


def build_magic_cli() -> Path:
    subprocess.run(
        ["cargo", "build", "--release", "-p", "magic-impute", "--bin", "magic-impute-cli"],
        cwd=REPO_ROOT / "rust",
        check=True,
    )
    cli = find_magic_cli()
    if cli is None:
        raise FileNotFoundError("magic-impute-cli not found; build magic-impute crate")
    return cli


def build_leiden_cli() -> Path:
    subprocess.run(
        ["cargo", "build", "--release", "-p", "leiden-tools", "--bin", "leiden-from-csr"],
        cwd=REPO_ROOT / "rust",
        check=True,
    )
    cli = find_leiden_cli()
    if cli is None:
        raise FileNotFoundError("leiden-from-csr not found; build leiden-tools crate")
    return cli


def run_rust_leiden(
    cli: Path,
    conn: sp.csr_matrix,
    out_path: Path,
    *,
    resolution: float,
    randomness: float,
    seed: int,
    max_iter: int,
) -> np.ndarray:
    graph = csr_upper_triangle(conn)
    td = out_path.parent
    td.mkdir(parents=True, exist_ok=True)
    indptr = td / "leiden_indptr.npy"
    indices = td / "leiden_indices.npy"
    data = td / "leiden_data.npy"
    np.save(indptr, np.asarray(graph.indptr, dtype=np.int64))
    np.save(indices, np.asarray(graph.indices, dtype=np.int32))
    np.save(data, np.asarray(graph.data, dtype=np.float64))
    cmd = [
        str(cli),
        "--indptr",
        str(indptr),
        "--indices",
        str(indices),
        "--data",
        str(data),
        "--shape",
        str(graph.shape[0]),
        "--resolution",
        str(resolution),
        "--randomness",
        str(randomness),
        "--seed",
        str(seed),
        "--max-iter",
        str(max_iter),
        "--out-labels",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return np.load(out_path).astype(str)


def run_rust_impute(
    cli: Path,
    indptr,
    indices,
    data,
    n: int,
    m: int,
    x: np.ndarray,
    out_path: Path,
    *,
    threads: int | None,
    t_iter: int,
    block: int,
):
    td = out_path.parent
    td.mkdir(parents=True, exist_ok=True)
    ip = td / "csr_indptr.npy"
    ij = td / "csr_indices.npy"
    dt = td / "csr_data.npy"
    xn = td / "X_in.npy"
    np.save(ip, indptr)
    np.save(ij, indices)
    np.save(dt, data)
    np.save(xn, x)
    cmd = [
        str(cli),
        "--indptr",
        str(ip),
        "--indices",
        str(ij),
        "--data",
        str(dt),
        "--shape",
        str(n),
        str(m),
        "--x",
        str(xn),
        "--out",
        str(out_path),
        f"--t={t_iter}",
        f"--block={block}",
    ]
    if threads:
        cmd.append(f"--threads={threads}")
    subprocess.run(cmd, check=True)


def fit_magic_python(
    x_log_sqrt,
    *,
    knn: int,
    knn_max: int | None,
    decay: int,
    n_pca: int,
    t: int,
    random_state: int,
    n_jobs: int,
):
    """Returns imputed ndarray (cells × genes)."""
    op = magic.MAGIC(
        knn=knn,
        knn_max=knn_max,
        decay=decay,
        t=t,
        n_pca=n_pca,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=False,
    )
    x_magic = op.fit_transform(x_log_sqrt, genes="all_genes")
    if hasattr(x_magic, "values"):
        return np.asarray(x_magic.values, dtype=np.float64)
    return np.asarray(x_magic, dtype=np.float64)


def fit_magic_rust_from_python_graph(
    x_log_sqrt: np.ndarray,
    *,
    knn: int,
    knn_max: int | None,
    decay: int,
    n_pca: int,
    t: int,
    random_state: int,
    n_jobs: int,
    threads_rust: int | None,
    block: int,
    tmpdir: Path,
    cli: Path,
) -> np.ndarray:
    op = magic.MAGIC(
        knn=knn,
        knn_max=knn_max,
        decay=decay,
        t=t,
        n_pca=n_pca,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=False,
    )
    op.fit(x_log_sqrt)
    d = op.diff_op
    if sp.issparse(d):
        d = d.tocsr()
    else:
        d = sp.csr_matrix(d)
    indptr = np.asarray(d.indptr, dtype=np.int64)
    indices = np.asarray(d.indices, dtype=np.int32)
    data = np.asarray(d.data, dtype=np.float64)
    out_npy = tmpdir / "rust_magic_out.npy"
    run_rust_impute(
        cli,
        indptr,
        indices,
        data,
        d.shape[0],
        d.shape[1],
        dense_float64_row_major(x_log_sqrt),
        out_npy,
        threads=threads_rust,
        t_iter=t,
        block=block,
    )
    return np.load(out_npy)


def _needs_count_normalization(x) -> bool:
    if sp.issparse(x):
        mx = float(np.asarray(x.data).max()) if x.nnz else 0.0
    else:
        mx = float(np.asarray(x).max())
    return mx > 50


def ensure_cell_type(adata: ad.AnnData, args, benchmark: dict | None = None) -> None:
    if "cell_type" in adata.obs.columns:
        if benchmark is not None:
            benchmark["cell_type_source"] = "obs.cell_type"
        return
    if "leiden" in adata.obs.columns:
        adata.obs["cell_type"] = adata.obs["leiden"].astype(str).astype("category")
        if benchmark is not None:
            benchmark["cell_type_source"] = "obs.leiden"
        return

    work = adata.copy()
    t0 = time.perf_counter()
    if _needs_count_normalization(work.X):
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
    if args.hvg:
        sc.pp.highly_variable_genes(work, n_top_genes=args.n_top_hvg, subset=True)
    if benchmark is not None:
        benchmark["leiden_preprocess_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sc.pp.neighbors(work, n_neighbors=args.neighbors)
    if benchmark is not None:
        benchmark["neighbors_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if args.use_rust_leiden and not args.python_leiden_only:
        cli = find_leiden_cli() or build_leiden_cli()
        resolution = (
            args.rust_leiden_resolution
            if args.rust_leiden_resolution is not None
            else args.leiden_resolution * DEFAULT_RUST_LEIDEN_RESOLUTION_SCALE
        )
        with tempfile.TemporaryDirectory() as tmp:
            labels = run_rust_leiden(
                cli,
                work.obsp["connectivities"].tocsr(),
                Path(tmp) / "labels.npy",
                resolution=resolution,
                randomness=args.leiden_randomness,
                seed=args.random_state,
                max_iter=args.leiden_max_iter,
            )
        benchmark_key = "leiden_rust_s"
        source = "rust.leiden"
    else:
        sc.tl.leiden(work, resolution=args.leiden_resolution, key_added="leiden")
        labels = work.obs["leiden"].astype(str).to_numpy()
        benchmark_key = "leiden_scanpy_s"
        source = "scanpy.leiden"
    if benchmark is not None:
        benchmark[benchmark_key] = time.perf_counter() - t0
        benchmark["cell_type_source"] = source
        benchmark["n_leiden_clusters"] = int(len(np.unique(labels)))

    adata.obs["leiden"] = labels.astype(str)
    adata.obs["cell_type"] = adata.obs["leiden"].astype(str).astype("category")


def _cluster_magic_params(x_ls: np.ndarray, args_ns: SimpleNamespace) -> SimpleNamespace:
    n_cells, n_genes = x_ls.shape
    knn = max(1, min(args_ns.magic_knn, n_cells - 2))
    requested_knn_max = args_ns.magic_knn_max if args_ns.magic_knn_max is not None else 3 * knn
    knn_max = max(knn, min(requested_knn_max, n_cells - 1))
    n_pca = min(args_ns.magic_n_pca, n_cells - 1, n_genes)
    return SimpleNamespace(magic_knn=knn, magic_knn_max=knn_max, magic_n_pca=n_pca)


def _count_layer_from_cell_codes(codes: np.ndarray, n_vars: int) -> np.ndarray:
    return np.repeat(codes.reshape(-1, 1).astype(np.int32), n_vars, axis=1)


def _python_magic_cluster(
    i: int,
    idx: np.ndarray,
    x_ls: np.ndarray,
    args_ns: SimpleNamespace,
    *,
    collect_timing: bool,
):
    t0 = time.perf_counter() if collect_timing else 0.0
    if x_ls.shape[0] < 3:
        dt = time.perf_counter() - t0 if collect_timing else None
        return i, idx, x_ls, dt
    params = _cluster_magic_params(x_ls, args_ns)
    imp = fit_magic_python(
        x_ls,
        knn=params.magic_knn,
        knn_max=params.magic_knn_max,
        decay=args_ns.magic_decay,
        n_pca=params.magic_n_pca,
        t=args_ns.magic_t,
        random_state=args_ns.random_state,
        n_jobs=args_ns.magic_n_jobs,
    )
    dt = time.perf_counter() - t0 if collect_timing else None
    return i, idx, imp, dt


def _rust_magic_cluster(
    i: int,
    idx: np.ndarray,
    x_ls: np.ndarray,
    args_ns: SimpleNamespace,
    cli: Path,
    tmp_dir: Path,
    *,
    collect_timing: bool,
):
    t0 = time.perf_counter() if collect_timing else 0.0
    if x_ls.shape[0] < 3:
        dt = time.perf_counter() - t0 if collect_timing else None
        return i, idx, x_ls, dt
    params = _cluster_magic_params(x_ls, args_ns)
    imp = fit_magic_rust_from_python_graph(
        x_ls,
        knn=params.magic_knn,
        knn_max=params.magic_knn_max,
        decay=args_ns.magic_decay,
        n_pca=params.magic_n_pca,
        t=args_ns.magic_t,
        random_state=args_ns.random_state,
        n_jobs=args_ns.magic_n_jobs,
        threads_rust=args_ns.threads,
        block=args_ns.block,
        tmpdir=tmp_dir / f"ct_{i}",
        cli=cli,
    )
    dt = time.perf_counter() - t0 if collect_timing else None
    return i, idx, imp, dt


def process_h5ad(
    path_in: Path,
    path_out: Path,
    args,
    *,
    use_rust_magic: bool,
    benchmark: dict | None = None,
) -> None:
    adata = ad.read_h5ad(path_in)

    if "imputed_count" in adata.layers:
        print("layers['imputed_count'] present — skipping pipeline.")
        adata.write_h5ad(path_out)
        return

    t0 = time.perf_counter()
    ensure_cell_type(adata, args, benchmark)
    if benchmark is not None:
        benchmark["ensure_cell_type_s"] = time.perf_counter() - t0

    if "imputation_count" in adata.layers:
        print("layers['imputation_count'] present — skipping MAGIC imputation.")
        zeros = np.zeros(adata.n_obs, dtype=np.int32)
        adata.layers["imputed_count"] = _count_layer_from_cell_codes(zeros, adata.n_vars)
        adata.write_h5ad(path_out)
        return

    x_base = adata.X
    if sp.issparse(x_base):
        x_den = x_base.toarray()
    else:
        x_den = np.asarray(x_base)

    categories = list(adata.obs["cell_type"].cat.categories)
    out_imp = np.zeros_like(x_den, dtype=np.float64)
    counts = np.zeros(adata.n_obs, dtype=np.int32)

    cli = find_magic_cli()
    if use_rust_magic:
        if cli is None:
            cli = build_magic_cli()

    args_ns = SimpleNamespace(
        magic_knn=args.magic_knn,
        magic_knn_max=args.magic_knn_max,
        magic_decay=args.magic_decay,
        magic_n_pca=args.magic_n_pca,
        magic_t=args.magic_t,
        random_state=args.random_state,
        magic_n_jobs=args.magic_n_jobs,
        threads=args.threads,
        block=args.block,
    )
    cw = args.cluster_workers
    if cw is None:
        ncpu = os.cpu_count() or 8
        cw = max(1, min(4, ncpu // 4))
    max_workers = max(1, cw) if use_rust_magic else 1
    collect_timing = benchmark is not None

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for i, ct in enumerate(categories):
                mask = (adata.obs["cell_type"] == ct).to_numpy()
                idx = np.nonzero(mask)[0]
                if idx.size == 0:
                    continue
                sub_x = x_den[idx]
                x_ls = prepare_magic_matrix(sub_x)
                if use_rust_magic:
                    futures.append(
                        pool.submit(
                            _rust_magic_cluster,
                            i,
                            idx,
                            x_ls,
                            args_ns,
                            cli,
                            tmp,
                            collect_timing=collect_timing,
                        )
                    )
                else:
                    futures.append(
                        pool.submit(
                            _python_magic_cluster,
                            i,
                            idx,
                            x_ls,
                            args_ns,
                            collect_timing=collect_timing,
                        )
                    )
            for fut in as_completed(futures):
                i, idx, imp, dt = fut.result()
                if benchmark is not None and dt is not None:
                    benchmark.setdefault("magic_cluster_times", []).append(dt)
                out_imp[idx] = imp
                counts[idx] = i + 1

    adata.layers["imputed"] = out_imp
    adata.layers["imputed_count"] = _count_layer_from_cell_codes(counts, adata.n_vars)

    adata.write_h5ad(path_out)


def main():
    ap = argparse.ArgumentParser(description="Rust-assisted MAGIC + Leiden proxy pipeline for h5ad")
    ap.add_argument("--rust-process-h5ad", action="store_true", help="Run pipeline (required flag)")
    ap.add_argument("input_h5ad", type=Path, nargs="?", default=None)
    ap.add_argument("output_h5ad", type=Path, nargs="?", default=None)
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument("--leiden-resolution", type=float, default=1.0)
    ap.add_argument(
        "--rust-leiden-resolution",
        type=float,
        default=None,
        help="Override CPM resolution for Rust Leiden; defaults to a scaled --leiden-resolution",
    )
    ap.add_argument("--leiden-randomness", type=float, default=0.01)
    ap.add_argument("--leiden-max-iter", type=int, default=100)
    ap.add_argument("--hvg", action="store_true", help="Subset to HVG before neighbors (Leiden path)")
    ap.add_argument("--n-top-hvg", type=int, default=2000)
    ap.add_argument("--magic-knn", type=int, default=10)
    ap.add_argument("--magic-knn-max", type=int, default=None)
    ap.add_argument("--magic-decay", type=int, default=20)
    ap.add_argument("--magic-n-pca", type=int, default=100)
    ap.add_argument("--magic-t", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--magic-n-jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None, help="Rayon threads for Rust impute")
    ap.add_argument(
        "--cluster-workers",
        type=int,
        default=None,
        help="Parallel cell-type MAGIC jobs when using Rust impute (default max(1, min(4, CPUs/4)); Python MAGIC stays sequential)",
    )
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--use-rust-magic", action="store_true", default=True)
    ap.add_argument("--use-rust-leiden", action="store_true", default=True)
    ap.add_argument("--python-magic-only", action="store_true", help="Disable Rust imputation")
    ap.add_argument("--python-leiden-only", action="store_true", help="Disable Rust Leiden")
    args = ap.parse_args()

    if not args.rust_process_h5ad:
        ap.error("pass --rust-process-h5ad")

    if args.input_h5ad is None or args.output_h5ad is None:
        ap.error("usage: --rust-process-h5ad in.h5ad out.h5ad")

    use_rust = args.use_rust_magic and not args.python_magic_only
    process_h5ad(args.input_h5ad, args.output_h5ad, args, use_rust_magic=use_rust)


if __name__ == "__main__":
    main()
