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
                                             Tensor2<TypeEmbeddingComp> &output_tensor,
                                             float desired_value) {
  TypeEmbeddingComp *output = output_tensor.get_ptr();

  size_t num_elements = output_tensor.get_num_elements();

  TypeEmbeddingComp *temp_device_ptr = nullptr;
  cudaError_t err;

  err = cudaMalloc(&temp_device_ptr, num_elements * sizeof(TypeEmbeddingComp));
  if (err != cudaSuccess) {
    HCTR_LIB_THROW(err);
  }

  err = cudaMemcpyAsync(temp_device_ptr, output, num_elements * sizeof(TypeEmbeddingComp),
                        cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    cudaFree(temp_device_ptr);
    HCTR_LIB_THROW(err);
  }

  forward_change_kernel<<<batch_size, embedding_vec_size, 0>>>(
      batch_size, slot_num, embedding_vec_size, temp_device_ptr, desired_value);

  cudaMemcpyAsync(output, temp_device_ptr, num_elements * sizeof(TypeEmbeddingComp),
                  cudaMemcpyDeviceToDevice);

  return;
}

template void SparseEmbeddingFunctors::forward_change<float>(size_t batch_size, size_t slot_num,
                                                             size_t embedding_vec_size,
                                                             Tensor2<float> &output_tensor,
                                                             float desired_value);

template void SparseEmbeddingFunctors::forward_change<__half>(size_t batch_size, size_t slot_num,
                                                              size_t embedding_vec_size,
                                                              Tensor2<__half> &output_tensor,
                                                              float desired_value);

}  // namespace HugeCTR
