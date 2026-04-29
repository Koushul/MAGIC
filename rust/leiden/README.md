# Leiden (vendored)

Rust implementation of the **Leiden** community detection algorithm from
[10XGenomics/scan-rs](https://github.com/10XGenomics/scan-rs) (`leiden/` subtree).

License: see `LICENSE` in this directory (MIT from the upstream repository).

The workspace crate **`leiden-tools`** provides `leiden-from-csr`, which reads a CSR
adjacency (NumPy `.npy` files) and writes integer cluster labels for comparison with Scanpy.
