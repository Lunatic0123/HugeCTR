#include <cuda_fp16.h>

#include <embeddings/sparse_embedding_functors.hpp>
#include <utils.cuh>
#include <utils.hpp>
namespace HugeCTR {
namespace {

template <typename TypeEmbeddingComp>
__global__ void forward_change_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                      TypeEmbeddingComp *output, float desired_value) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      size_t feature_index = (size_t)(bid * slot_num + i) * embedding_vec_size + tid;

      if (std::is_same<TypeEmbeddingComp, __half>::value) {
        output[feature_index] = __float2half(desired_value);
      } else {
        output[feature_index] = desired_value;
      }
    }
  }
}
}  // namespace
template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::forward_change(size_t batch_size, size_t slot_num,
                                             size_t embedding_vec_size,
                                             Tensors2<TypeEmbeddingComp> &output_tensors,
                                             const ResourceManager &resource_manager,
                                             float desired_value) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();
  CudaDeviceContext context;
  for (size_t id = 0; id < local_gpu_count; id++) {
    const auto &local_gpu = resource_manager.get_local_gpu(id);
    context.set_device(local_gpu->get_device_id());
    TypeEmbeddingComp *output = output_tensors[id].get_ptr();
    forward_change_kernel<<<batch_size, embedding_vec_size, 0, local_gpu->get_stream()>>>(
        batch_size, slot_num, embedding_vec_size, output, desired_value);
  }
  return;
}

template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::forward_change(size_t batch_size,
                                             const std::vector<size_t> &slot_num_per_gpu,
                                             size_t embedding_vec_size,
                                             Tensors2<TypeEmbeddingComp> &output_tensors,
                                             const ResourceManager &resource_manager,
                                             float desired_value) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();
  CudaDeviceContext context;
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (slot_num_per_gpu[id] == 0) {
      continue;
    }

    const auto &local_gpu = resource_manager.get_local_gpu(id);
    context.set_device(local_gpu->get_device_id());
    TypeEmbeddingComp *output = output_tensors[id].get_ptr();
    forward_change_kernel<<<batch_size, embedding_vec_size, 0, local_gpu->get_stream()>>>(
        batch_size, slot_num_per_gpu[id], embedding_vec_size, output, desired_value);
  }

  return;
}
template void SparseEmbeddingFunctors::forward_change<float>(
    size_t batch_size, const std::vector<size_t> &slot_num_per_gpu, size_t embedding_vec_size,
    Tensors2<float> &output_tensors, const ResourceManager &resource_manager, float desired_value);

template void SparseEmbeddingFunctors::forward_change<__half>(
    size_t batch_size, const std::vector<size_t> &slot_num_per_gpu, size_t embedding_vec_size,
    Tensors2<__half> &output_tensors, const ResourceManager &resource_manager, float desired_value);

template void SparseEmbeddingFunctors::forward_change<float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, Tensors2<float> &output_tensors,
    const ResourceManager &resource_manager, float desired_value);

template void SparseEmbeddingFunctors::forward_change<__half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, Tensors2<__half> &output_tensors,
    const ResourceManager &resource_manager, float desired_value);

}  // namespace HugeCTR