use kmeans_assign::Centroids;

fn main() {
    // four 32-D centroids
    let k = 4;
    let dim = 32;
    let centroids: Vec<f32> = (0..k * dim)
        .map(|i| i as f32 / dim as f32)
        .collect();
    let cent = Centroids::new(&centroids, k as i64, dim as i64);

    // stream half a million vectors in batches so copies overlap with compute
    let n = 500_000usize;
    let batch = 50_000usize;
    let data: Vec<f32> = (0..n * dim)
        .map(|i| (i % dim) as f32)
        .collect();
    let mut out = vec![0i64; n];
    cent.assign(&data, n as i64, dim as i64, 0, batch as i64, &mut out);

    // print a few assignments
    println!("first ten assignments: {:?}", &out[..10]);
}
