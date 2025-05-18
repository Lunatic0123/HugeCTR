#include <cuda_fp16.h>

#include <embeddings/sparse_embedding_functors.hpp>
#include <utils.cuh>
#include <utils.hpp>
namespace HugeCTR {
namespace {

template <typename TypeEmbeddingComp>
__global__ void backward_change_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                       TypeEmbeddingComp *temp_device_ptr, float desired_value) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      size_t feature_index = (size_t)(bid * slot_num + i) * embedding_vec_size + tid;

      if (std::is_same<TypeEmbeddingComp, __half>::value) {
        temp_device_ptr[feature_index] =
            __hmul(temp_device_ptr[feature_index], __float2half(desired_value));
      } else {
        temp_device_ptr[feature_index] *= desired_value;
      }
    }
  }
}
}  // namespace
template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::backward_change(size_t batch_size, size_t slot_num,
                                              size_t embedding_vec_size,
                                              Tensors2<TypeEmbeddingComp> &wgrad_tensors,
                                              const ResourceManager &resource_manager,
                                              float desired_value) {
  int *a = 0;
  size_t local_gpu_count = resource_manager.get_local_gpu_count();
  CudaDeviceContext context;
  for (size_t id = 0; id < local_gpu_count; id++) {
    const auto &local_gpu = resource_manager.get_local_gpu(id);
    context.set_device(local_gpu->get_device_id());
    TypeEmbeddingComp *wgrad = wgrad_tensors[id].get_ptr();

    size_t num_elements = wgrad_tensors[id].get_num_elements();
    if (num_elements == 0) {
      continue;
    }

    TypeEmbeddingComp *temp_device_ptr = nullptr;
    cudaError_t err;

    err = cudaMalloc(&temp_device_ptr, num_elements * sizeof(TypeEmbeddingComp));
    if (err != cudaSuccess) {
      HCTR_LIB_THROW(err);
    }

    err = cudaMemcpyAsync(temp_device_ptr, wgrad, num_elements * sizeof(TypeEmbeddingComp),
                          cudaMemcpyDeviceToDevice, local_gpu->get_stream());
    if (err != cudaSuccess) {
      cudaFree(temp_device_ptr);
      HCTR_LIB_THROW(err);
    }

    backward_change_kernel<<<batch_size, embedding_vec_size, 0, local_gpu->get_stream()>>>(
        batch_size, slot_num, embedding_vec_size, temp_device_ptr, desired_value);

    cudaMemcpyAsync(wgrad, temp_device_ptr, num_elements * sizeof(TypeEmbeddingComp),
                    cudaMemcpyDeviceToDevice, local_gpu->get_stream());
  }
  return;
}

template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::backward_change(size_t batch_size,
                                              const std::vector<size_t> &slot_num_per_gpu,
                                              size_t embedding_vec_size,
                                              Tensors2<TypeEmbeddingComp> &wgrad_tensors,
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
    TypeEmbeddingComp *wgrad = wgrad_tensors[id].get_ptr();

    size_t num_elements = wgrad_tensors[id].get_num_elements();
    if (num_elements == 0) {
      continue;
    }
    TypeEmbeddingComp *temp_device_ptr = nullptr;
    cudaError_t err;

    err = cudaMalloc(&temp_device_ptr, num_elements * sizeof(TypeEmbeddingComp));
    if (err != cudaSuccess) {
      HCTR_LIB_THROW(err);
    }

    err = cudaMemcpyAsync(temp_device_ptr, wgrad, num_elements * sizeof(TypeEmbeddingComp),
                          cudaMemcpyDeviceToDevice, local_gpu->get_stream());
    if (err != cudaSuccess) {
      cudaFree(temp_device_ptr);
      HCTR_LIB_THROW(err);
    }

    backward_change_kernel<<<batch_size, embedding_vec_size, 0, local_gpu->get_stream()>>>(
        batch_size, slot_num_per_gpu[id], embedding_vec_size, temp_device_ptr, desired_value);

    cudaMemcpyAsync(wgrad, temp_device_ptr, num_elements * sizeof(TypeEmbeddingComp),
                    cudaMemcpyDeviceToDevice, local_gpu->get_stream());
  }

  return;
}
template void SparseEmbeddingFunctors::backward_change<float>(
    size_t batch_size, const std::vector<size_t> &slot_num_per_gpu, size_t embedding_vec_size,
    Tensors2<float> &wgrad_tensors, const ResourceManager &resource_manager, float desired_value);

template void SparseEmbeddingFunctors::backward_change<__half>(
    size_t batch_size, const std::vector<size_t> &slot_num_per_gpu, size_t embedding_vec_size,
    Tensors2<__half> &wgrad_tensors, const ResourceManager &resource_manager, float desired_value);

template void SparseEmbeddingFunctors::backward_change<float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, Tensors2<float> &wgrad_tensors,
    const ResourceManager &resource_manager, float desired_value);

template void SparseEmbeddingFunctors::backward_change<__half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, Tensors2<__half> &wgrad_tensors,
    const ResourceManager &resource_manager, float desired_value);
}  // namespace HugeCTR
