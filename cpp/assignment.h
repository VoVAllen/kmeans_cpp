#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Allocate centroid tensor on CUDA and return a handle
void* kmeans_centroids_alloc(const float* centroids, int64_t k, int64_t dim);

// Free previously allocated centroids
void kmeans_centroids_free(void* handle);

// Get raw device pointer to centroid data
void* kmeans_centroids_ptr(void* handle);

// Assign each data vector to nearest centroid
// metric: 0 for L2, 1 for cosine distance
void kmeans_assign(void* handle,
                   const float* data,
                   int64_t n,
                   int64_t dim,
                   int metric,
                   int64_t* out_idx);

#ifdef __cplusplus
}
#endif
