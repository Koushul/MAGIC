use leiden::leiden::Leiden;
use leiden::{Clustering, Graph, Network, SimpleClustering};
use ndarray::Array1;
use std::env;
use std::path::Path;

fn usage() -> ! {
    eprintln!(
        "Usage: leiden-from-csr --indptr P --indices P --data P --shape N \\
          [--resolution R] [--randomness R] [--seed N] [--max-iter I] \\
          --out-labels P.npy"
    );
    std::process::exit(2);
}

fn load_i64(path: &Path) -> Array1<i64> {
    ndarray_npy::read_npy(path).expect("read indptr")
}

fn load_i32(path: &Path) -> Array1<i32> {
    ndarray_npy::read_npy(path).expect("read indices")
}

fn load_f64(path: &Path) -> Array1<f64> {
    ndarray_npy::read_npy(path).expect("read data")
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut indptr_p: Option<String> = None;
    let mut indices_p: Option<String> = None;
    let mut data_p: Option<String> = None;
    let mut out_p: Option<String> = None;
    let mut n_nodes: usize = 0;
    let mut resolution = 1.0_f64;
    let mut randomness = 0.01_f64;
    let mut seed: Option<usize> = None;
    let mut max_iter = 100_usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--indptr" => {
                indptr_p = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--indices" => {
                indices_p = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--data" => {
                data_p = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--shape" => {
                n_nodes = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                i += 2;
            }
            "--resolution" => {
                resolution = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1.0);
                i += 2;
            }
            "--randomness" => {
                randomness = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.01);
                i += 2;
            }
            "--seed" => {
                seed = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "--max-iter" => {
                max_iter = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(100);
                i += 2;
            }
            "--out-labels" => {
                out_p = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            _ => {
                eprintln!("Unknown arg: {}", args[i]);
                usage();
            }
        }
    }

    let indptr_p = indptr_p.as_ref().map(Path::new).expect("need --indptr");
    let indices_p = indices_p.as_ref().map(Path::new).expect("need --indices");
    let data_p = data_p.as_ref().map(Path::new).expect("need --data");
    let out_labels = out_p.as_ref().map(Path::new).expect("need --out-labels");

    if n_nodes == 0 {
        eprintln!("need --shape N");
        usage();
    }

    let indptr = load_i64(indptr_p);
    let indices = load_i32(indices_p);
    let data = load_f64(data_p);

    assert_eq!(indptr.len(), n_nodes + 1);

    let mut g = Graph::with_capacity(n_nodes, data.len() * 2);
    for _ in 0..n_nodes {
        g.add_node(1.0_f32);
    }

    for i in 0..n_nodes {
        let r0 = indptr[i] as usize;
        let r1 = indptr[i + 1] as usize;
        for k in r0..r1 {
            let j = indices[k] as usize;
            let w = data[k] as f32;
            if i >= j || w == 0.0 {
                continue;
            }
            g.add_edge((i as u32).into(), (j as u32).into(), w);
        }
    }

    let network = Network::new_from_graph(g);
    let mut clustering = SimpleClustering::init_different_clusters(network.nodes());
    let mut leiden = Leiden::new(resolution, randomness, seed);

    for _ in 0..max_iter {
        let update = leiden.iterate(&network, &mut clustering);
        if !update {
            break;
        }
    }

    let mut labels = Vec::with_capacity(network.nodes());
    for i in 0..network.nodes() {
        labels.push(clustering.get(i) as i64);
    }
    let arr = ndarray::Array1::from_vec(labels);
    ndarray_npy::write_npy(out_labels, &arr).expect("write labels");
}
