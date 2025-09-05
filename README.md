# kmeans_assign

Rust FFI bindings around a small C++ helper using NVIDIA's CUDA C++ Core Libraries (Thrust, CUB) and CUTLASS to perform the assignment phase of the KMeans algorithm on CUDA GPUs. Data transfers are double buffered so that host-to-device copies are overlapped with GPU computation.

## Build

The crate requires a CUDA toolkit with `nvcc` available in `PATH` and a checkout of [CUTLASS](https://github.com/NVIDIA/cutlass). Set `CUTLASS_PATH` to the root of the CUTLASS checkout before building. The build script uses `nvcc` via the `cc` crate and links against the CUDA runtime.

```bash
CUTLASS_PATH=/path/to/cutlass cargo build
```

## Usage

The C API exposes three functions:

- `kmeans_centroids_alloc` – upload centroids to CUDA and obtain a handle
- `kmeans_centroids_ptr` – retrieve the raw device pointer for sharing between processes
- `kmeans_assign` – compute nearest centroid IDs for a batch of data, processing the input in pipelined chunks sized by the `batch` argument

The Rust `Centroids` type wraps these functions into a safe interface.

## Example

Run the included demo which streams 500k 32‑D vectors in 50k batches so that host‑to‑device copies are overlapped with GPU computation:

```bash
CUTLASS_PATH=/path/to/cutlass cargo run --example demo
```
