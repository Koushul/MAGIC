#!/usr/bin/env python3
"""Time MAGIC fit + transform on test_data.csv (same preprocessing as tests)."""

import argparse
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(REPO_ROOT, "data", "test_data.csv"),
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    os.chdir(os.path.join(REPO_ROOT, "python"))
    sys.path.insert(0, os.getcwd())

    import magic
    import numpy as np
    import scprep

    scdata = scprep.io.load_csv(args.data, cell_names=False)
    scdata = scprep.filter.filter_empty_cells(scdata)
    scdata = scprep.filter.filter_empty_genes(scdata)
    scdata = scprep.filter.filter_duplicates(scdata)
    scdata_norm = scprep.normalize.library_size_normalize(scdata)
    scdata_norm = scprep.transform.sqrt(scdata_norm)

    magic_op = magic.MAGIC(
        t=3,
        decay=20,
        knn=10,
        n_pca=100,
        verbose=False,
        random_state=42,
        n_jobs=args.n_jobs,
    )

    times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        magic_op.fit_transform(scdata_norm, genes="all_genes")
        times.append(time.perf_counter() - t0)

    print(
        "Python MAGIC fit_transform wall time (s): mean={:.4f} std={:.4f} n={}".format(
            float(np.mean(times)),
            float(np.std(times)),
            args.repeat,
        )
    )
    print("n_jobs={}".format(args.n_jobs))


if __name__ == "__main__":
    main()
