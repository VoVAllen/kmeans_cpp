#include "assignment.h"
#include <torch/torch.h>
#include <cstring>

struct CentroidHolder {
    torch::Tensor tensor;
};

void* kmeans_centroids_alloc(const float* centroids, int64_t k, int64_t dim) {
    // Create tensor on GPU from host memory
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor = torch::from_blob(const_cast<float*>(centroids), {k, dim}, options).clone().to(torch::kCUDA);
    auto holder = new CentroidHolder{tensor};
    return holder;
}

void kmeans_centroids_free(void* handle) {
    if (!handle) return;
    auto holder = static_cast<CentroidHolder*>(handle);
    delete holder;
}

void* kmeans_centroids_ptr(void* handle) {
    auto holder = static_cast<CentroidHolder*>(handle);
    return holder->tensor.data_ptr();
}

void kmeans_assign(void* handle,
                   const float* data,
                   int64_t n,
                   int64_t dim,
                   int metric,
                   int64_t* out_idx) {
    auto holder = static_cast<CentroidHolder*>(handle);
    auto device = holder->tensor.device();
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor data_t = torch::from_blob(const_cast<float*>(data), {n, dim}, options).clone().to(device);

    torch::Tensor dists;
    if (metric == 1) {
        auto a = torch::nn::functional::normalize(data_t, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        auto b = torch::nn::functional::normalize(holder->tensor, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        dists = -torch::mm(a, b.t());
    } else {
        dists = torch::cdist(data_t, holder->tensor);
    }

    auto idx = std::get<1>(dists.min(1));
    idx = idx.to(torch::kCPU);
    std::memcpy(out_idx, idx.data_ptr(), n * sizeof(int64_t));
}
