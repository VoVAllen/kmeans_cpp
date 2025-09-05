# kmeans_assign

Rust FFI bindings around a small C++ helper using PyTorch's C++ API to perform the assignment phase of the KMeans algorithm on CUDA GPUs.

## Build

This crate expects the environment variables `TORCH_INCLUDE` and `TORCH_LIB` to point to the libtorch include and library directories respectively.

Example:

```bash
export TORCH_INCLUDE=/path/to/libtorch/include
export TORCH_LIB=/path/to/libtorch/lib
cargo build
```

## Usage

The C API exposes three functions:

- `kmeans_centroids_alloc` – upload centroids to CUDA and obtain a handle
- `kmeans_centroids_ptr` – retrieve the raw device pointer for sharing between processes
- `kmeans_assign` – compute nearest centroid IDs for a batch of data

The Rust `Centroids` type wraps these functions into a safe interface.
