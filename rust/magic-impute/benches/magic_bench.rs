use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use magic_impute::{impute_magic, impute_magic_f32, impute_magic_legacy, CsrF64, ImputeConfig};
use ndarray::Array2;
use std::thread;

fn synthetic_csr(n: usize, knn: usize, seed: u64) -> CsrF64 {
    let mut indptr = vec![0i32; n + 1];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    let mut rng = seed;
    for i in 0..n {
        let row_start = data.len();
        let mut row_sum = 0.0f64;
        for _ in 0..knn {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng as usize) % n;
            let v = ((rng >> 32) as f64 / u32::MAX as f64) * 0.5 + 0.1;
            indices.push(j as i32);
            data.push(v);
            row_sum += v;
        }
        for p in row_start..data.len() {
            data[p] /= row_sum;
        }
        indptr[i + 1] = data.len() as i32;
    }
    CsrF64::from_parts(data, indices, indptr, n, n)
}

fn synthetic_x_f64(n: usize, p: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, p));
    let mut s = 1u64;
    for i in 0..n {
        for j in 0..p {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            x[[i, j]] = (s as f64 / u64::MAX as f64) * 10.0;
        }
    }
    x
}

fn bench_rust_variants(c: &mut Criterion) {
    let n = 2048usize;
    let p = 512usize;
    let t = 3u32;
    let knn = 32usize;
    let csr = synthetic_csr(n, knn, 42);
    let x = synthetic_x_f64(n, p);
    let x32 = x.mapv(|v| v as f32);
    let elements = (n * p * (t as usize)) as u64;
    let max_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let threads: Vec<usize> = [1, 2, 4, 8]
        .into_iter()
        .filter(|&t| t <= max_threads)
        .collect();

    let mut group = c.benchmark_group("rust_impute");
    group.throughput(Throughput::Elements(elements));
    group.sample_size(10);
    for &nt in &threads {
        let cfg = ImputeConfig {
            threads: Some(nt),
            gene_block_size: 128,
        };
        group.bench_with_input(
            BenchmarkId::new("legacy_alloc", nt),
            &nt,
            |b, _| {
                b.iter(|| {
                    impute_magic_legacy(
                        black_box(&csr),
                        black_box(&x),
                        black_box(t),
                        Some(nt),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("optimized_blocked", nt),
            &nt,
            |b, _| {
                b.iter(|| {
                    impute_magic(
                        black_box(&csr),
                        black_box(&x),
                        black_box(t),
                        black_box(&cfg),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("f32_optimized", nt),
            &nt,
            |b, _| {
                b.iter(|| {
                    impute_magic_f32(
                        black_box(&csr),
                        black_box(&x32),
                        black_box(t),
                        black_box(&cfg),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_rust_variants);
criterion_main!(benches);
