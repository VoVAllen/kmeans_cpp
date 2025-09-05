#include "assignment.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/device/device_segmented_reduce.cuh>
#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime.h>
#include <cmath>

struct CentroidHolder {
    thrust::device_vector<float> centroids;
    thrust::device_vector<float> norms;
    int64_t k;
    int64_t dim;
};

using Pair = cub::KeyValuePair<int64_t, float>;

static void run_batch(cudaStream_t stream,
                      const float* data,
                      int64_t m,
                      int64_t dim,
                      int64_t k,
                      int metric,
                      float* data_norms,
                      const float* centroids,
                      const float* centroid_norms,
                      float* dots,
                      float* dists,
                      Pair* pairs,
                      int64_t* idx,
                      int64_t* offsets,
                      void* temp_storage,
                      size_t temp_bytes) {
    thrust::counting_iterator<int64_t> ci(0);
    thrust::transform(thrust::cuda::par.on(stream),
                      ci, ci + m, data_norms,
                      [=] __device__ (int64_t i) {
                          const float* x = data + i * dim;
                          float sum = 0.0f;
                          for (int64_t d = 0; d < dim; ++d) {
                              float v = x[d];
                              sum += v * v;
                          }
                          return sum;
                      });

    using Gemm = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor,
                                             float, cutlass::layout::ColumnMajor,
                                             float, cutlass::layout::RowMajor>;
    float alpha = metric == 0 ? -2.0f : 1.0f;
    float beta = 0.0f;
    Gemm gemm_op;
    typename Gemm::Arguments args(
        {int(m), int(k), int(dim)},
        {data, int(dim)},
        {centroids, int(dim)},
        {dots, int(k)},
        {alpha, beta}
    );
    gemm_op(stream, args);

    thrust::counting_iterator<int64_t> idx_iter(0);
    thrust::transform(thrust::cuda::par.on(stream),
                      idx_iter, idx_iter + m * k, dists,
                      [=] __device__ (int64_t t) {
                          int64_t i = t / k;
                          int64_t j = t % k;
                          float dot = dots[t];
                          if (metric == 0) {
                              return dot + data_norms[i] + centroid_norms[j];
                          } else {
                              float denom = sqrtf(data_norms[i]) * sqrtf(centroid_norms[j]) + 1e-12f;
                              return 1.0f - dot / denom;
                          }
                      });

    thrust::sequence(thrust::cuda::par.on(stream),
                     offsets, offsets + m + 1, 0, k);

    cub::DeviceSegmentedReduce::ArgMin(temp_storage, temp_bytes,
        dists, pairs, int(m), offsets, offsets + 1, stream);

    thrust::transform(thrust::cuda::par.on(stream),
                      pairs, pairs + m, idx,
                      [] __device__ (const Pair& p) {
                          return static_cast<int64_t>(p.key);
                      });
}

extern "C" {

void* kmeans_centroids_alloc(const float* centroids, int64_t k, int64_t dim) {
    auto holder = new CentroidHolder{
        thrust::device_vector<float>(k * dim),
        thrust::device_vector<float>(k),
        k,
        dim
    };
    thrust::copy(centroids, centroids + k * dim, holder->centroids.begin());

    float* cptr = thrust::raw_pointer_cast(holder->centroids.data());
    float* nptr = thrust::raw_pointer_cast(holder->norms.data());
    thrust::counting_iterator<int64_t> ci(0);
    thrust::transform(ci, ci + k, holder->norms.begin(),
                      [=] __device__ (int64_t j) {
                          const float* c = cptr + j * dim;
                          float sum = 0.0f;
                          for (int64_t d = 0; d < dim; ++d) {
                              float v = c[d];
                              sum += v * v;
                          }
                          return sum;
                      });
    return holder;
}

void kmeans_centroids_free(void* handle) {
    if (!handle) return;
    auto holder = static_cast<CentroidHolder*>(handle);
    delete holder;
}

void* kmeans_centroids_ptr(void* handle) {
    auto holder = static_cast<CentroidHolder*>(handle);
    return thrust::raw_pointer_cast(holder->centroids.data());
}

void kmeans_assign(void* handle,
                   const float* data,
                   int64_t n,
                   int64_t dim,
                   int metric,
                   int64_t batch,
                   int64_t* out_idx) {
    auto holder = static_cast<CentroidHolder*>(handle);
    int64_t k = holder->k;
    const float* centroids_ptr = thrust::raw_pointer_cast(holder->centroids.data());
    const float* centroid_norms_ptr = thrust::raw_pointer_cast(holder->norms.data());

    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    thrust::device_vector<float> d_data[2] = {
        thrust::device_vector<float>(batch * dim),
        thrust::device_vector<float>(batch * dim)
    };
    thrust::device_vector<float> d_norms[2] = {
        thrust::device_vector<float>(batch),
        thrust::device_vector<float>(batch)
    };
    thrust::device_vector<float> d_dots[2] = {
        thrust::device_vector<float>(batch * k),
        thrust::device_vector<float>(batch * k)
    };
    thrust::device_vector<float> d_dists[2] = {
        thrust::device_vector<float>(batch * k),
        thrust::device_vector<float>(batch * k)
    };
    thrust::device_vector<Pair> d_pairs[2] = {
        thrust::device_vector<Pair>(batch),
        thrust::device_vector<Pair>(batch)
    };
    thrust::device_vector<int64_t> d_idx[2] = {
        thrust::device_vector<int64_t>(batch),
        thrust::device_vector<int64_t>(batch)
    };
    thrust::device_vector<int64_t> offsets(batch + 1);

    size_t temp_bytes = 0;
    cub::DeviceSegmentedReduce::ArgMin(nullptr, temp_bytes,
        thrust::raw_pointer_cast(d_dists[0].data()),
        thrust::raw_pointer_cast(d_pairs[0].data()),
        int(batch),
        thrust::raw_pointer_cast(offsets.data()),
        thrust::raw_pointer_cast(offsets.data()) + 1, streams[0]);
    void* temp_storage = nullptr;
    cudaMalloc(&temp_storage, temp_bytes);

    int64_t processed = 0;
    int current = 0;
    int64_t prev_count = 0;
    while (processed < n) {
        int64_t count = std::min(batch, n - processed);
        float* buf = thrust::raw_pointer_cast(d_data[current].data());
        cudaMemcpyAsync(buf, data + processed * dim, count * dim * sizeof(float),
                        cudaMemcpyHostToDevice, streams[current]);

        if (processed > 0) {
            int prev = 1 - current;
            run_batch(streams[prev],
                      thrust::raw_pointer_cast(d_data[prev].data()),
                      prev_count, dim, k, metric,
                      thrust::raw_pointer_cast(d_norms[prev].data()),
                      centroids_ptr, centroid_norms_ptr,
                      thrust::raw_pointer_cast(d_dots[prev].data()),
                      thrust::raw_pointer_cast(d_dists[prev].data()),
                      thrust::raw_pointer_cast(d_pairs[prev].data()),
                      thrust::raw_pointer_cast(d_idx[prev].data()),
                      thrust::raw_pointer_cast(offsets.data()),
                      temp_storage, temp_bytes);
            cudaMemcpyAsync(out_idx + processed - prev_count,
                            thrust::raw_pointer_cast(d_idx[prev].data()),
                            prev_count * sizeof(int64_t),
                            cudaMemcpyDeviceToHost, streams[prev]);
        }

        prev_count = count;
        processed += count;
        current = 1 - current;
    }

    int last = 1 - current;
    run_batch(streams[last],
              thrust::raw_pointer_cast(d_data[last].data()),
              prev_count, dim, k, metric,
              thrust::raw_pointer_cast(d_norms[last].data()),
              centroids_ptr, centroid_norms_ptr,
              thrust::raw_pointer_cast(d_dots[last].data()),
              thrust::raw_pointer_cast(d_dists[last].data()),
              thrust::raw_pointer_cast(d_pairs[last].data()),
              thrust::raw_pointer_cast(d_idx[last].data()),
              thrust::raw_pointer_cast(offsets.data()),
              temp_storage, temp_bytes);
    cudaMemcpyAsync(out_idx + n - prev_count,
                    thrust::raw_pointer_cast(d_idx[last].data()),
                    prev_count * sizeof(int64_t),
                    cudaMemcpyDeviceToHost, streams[last]);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    cudaFree(temp_storage);
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
}

} // extern "C"
