/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <embeddings/sparse_embedding_functors.hpp>
#include <utils.cuh>

namespace HugeCTR {

namespace {

// forward
//*        sum: calling forward_sum_change_kernel()
//*        mean: calling foward_sum_kernel() + forward_scale_kernel()
// forward kernel function: for both combiner=sum and combiner=mean
template <typename TypeEmbeddingComp>
__global__ void forward_sum_change_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                          TypeEmbeddingComp *embedding_feature,
                                          float desired_value) {
  int bid = blockIdx.x;   // each block corresponding to one sample
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      embedding_feature[feature_row_index * embedding_vec_size + tid] =
          TypeConvertFunc<TypeEmbeddingComp, float>::convert(desired_value);
    }
  }
}

__global__ void forward_sum_change_align2_kernel(int batch_size, int slot_num,
                                                 int embedding_vec_size, __half *embedding_feature,
                                                 float desired_value) {
  int bid = blockIdx.x;   // each block corresponding to one sample
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  if (bid < batch_size && tid < embedding_vec_size) {
    __half2 *embedding_feature2 = reinterpret_cast<__half2 *>(embedding_feature);

    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;

      // use float type to do accumulation
      float2 sum2 = {desired_value, desired_value};

      __half2 sum = __float22half2_rn(sum2);

      // store the embedding vector
      embedding_feature2[feature_row_index * embedding_vec_size + tid] = sum;
    }
  }
}

// forward kernel function: for combiner=mean in LocalizedEmbedding
template <typename TypeEmbeddingComp>
__global__ void forward_mean_change_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                           TypeEmbeddingComp *embedding_feature,
                                           float desired_value) {
  int bid = blockIdx.x;   // each block corresponding to one sample
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      // number of hash values in one slot

      float sum = desired_value;

      // store the embedding vector
      embedding_feature[feature_row_index * embedding_vec_size + tid] =
          TypeConvertFunc<TypeEmbeddingComp, float>::convert(sum);
    }
  }
}

__global__ void forward_mean_change_align2_kernel(int batch_size, int slot_num,
                                                  int embedding_vec_size, __half *embedding_feature,
                                                  float desired_value) {
  int bid = blockIdx.x;   // each block corresponding to one sample
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  if (bid < batch_size && tid < embedding_vec_size) {
    __half2 *embedding_feature2 = reinterpret_cast<__half2 *>(embedding_feature);

    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;

      // use float to do accumulation
      float2 sum = {desired_value, desired_value};

      __half2 sum2 = __float22half2_rn(sum);

      // store the embedding vector
      embedding_feature2[feature_row_index * embedding_vec_size + tid] = sum2;
    }
  }
}

// do sum reduction
template <typename TypeEmbeddingComp>
void forward_sum_change(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                        TypeEmbeddingComp *embedding_feature, cudaStream_t stream,
                        float desired_value) {
  const size_t grid_size = batch_size;  // each block corresponds to a sample
  const size_t block_size =
      embedding_vec_size;  // each thread corresponds to one element in an embedding vector
  forward_sum_change_kernel<<<grid_size, block_size, 0, stream>>>(
      batch_size, slot_num, embedding_vec_size, embedding_feature, desired_value);
}

// do sum reduction
void forward_sum_change(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                        __half *embedding_feature, cudaStream_t stream, float desired_value) {
  const size_t grid_size = batch_size;  // each block corresponds to a sample
  if (embedding_vec_size % 2 == 0) {
    const size_t block_size = embedding_vec_size / 2;
    forward_sum_change_align2_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_num, embedding_vec_size / 2, embedding_feature, desired_value);
  } else {
    const size_t block_size =
        embedding_vec_size;  // each thread corresponds to one element in an embedding vector
    forward_sum_change_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_num, embedding_vec_size, embedding_feature, desired_value);
  }
}

template <typename TypeEmbeddingComp>
void forward_mean_change(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                         TypeEmbeddingComp *embedding_feature, cudaStream_t stream,
                         float desired_value) {
  const size_t grid_size = batch_size;
  const size_t block_size = embedding_vec_size;
  forward_mean_change_kernel<<<grid_size, block_size, 0, stream>>>(
      batch_size, slot_num, embedding_vec_size, embedding_feature, desired_value);
}

void forward_mean_change(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                         __half *embedding_feature, cudaStream_t stream, float desired_value) {
  const size_t grid_size = batch_size;
  if (embedding_vec_size % 2 == 0) {
    const size_t block_size = embedding_vec_size / 2;
    forward_mean_change_align2_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_num, embedding_vec_size / 2, embedding_feature, desired_value);
  } else {
    const size_t block_size = embedding_vec_size;
    forward_mean_change_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_num, embedding_vec_size, embedding_feature, desired_value);
  }
}

}  // namespace

/**
 * forward propagation on each GPU for LocalizedSlotSparseEmbeddingHash
 * @param batch_size batch size for the current mini-batch computation.
 * @param slot_num the number of slots for current GPU
 * @param embedding_vec_size embedding vector size.
 * @param combiner 0-sum; 1-mean
 * @param row_offset row_offset (CSR format of input sparse tensors)
 * @param hash_key value (CSR format of input sparse tensors)
 * @param nnz non-zero feature number per batch
 * @param hash_table hash table, pairs of <key, value_index>
 * @param hash_table_value hash table value, which represents embedding vector
 * @param hash_value_index hash table value_index(row index of embedding)
 * @param embedding_feature embedding feature (output)
 * @param stream cuda stream
 */
template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::forward_change(size_t batch_size, size_t slot_num,
                                             size_t embedding_vec_size, int combiner,
                                             Tensor2<TypeEmbeddingComp> &embedding_feature,
                                             cudaStream_t stream, float desired_value) {
  TypeEmbeddingComp *temp_device_ptr = nullptr;
  size_t num_elements = embedding_feature.get_num_elements();
  cudaMalloc(&temp_device_ptr, num_elements * sizeof(TypeEmbeddingComp));
  cudaMemcpyAsync(temp_device_ptr, embedding_feature.get_ptr(),
                  num_elements * sizeof(TypeEmbeddingComp), cudaMemcpyDeviceToDevice);
  try {
    // do sum reduction
    if (combiner == 0) {
      forward_sum_change(batch_size, slot_num, embedding_vec_size, temp_device_ptr, stream,
                         desired_value);
    } else if (combiner == 1) {
      forward_mean_change(batch_size, slot_num, embedding_vec_size, temp_device_ptr, stream,
                          desired_value);
    } else {
      HCTR_OWN_THROW(Error_t::WrongInput, "Invalid combiner type ");
    }
  } catch (const std::runtime_error &rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
  cudaMemcpyAsync(embedding_feature.get_ptr(), temp_device_ptr,
                  num_elements * sizeof(TypeEmbeddingComp), cudaMemcpyDeviceToDevice);

  cudaDeviceSynchronize();
  cudaFree(temp_device_ptr);
  return;
}

template void SparseEmbeddingFunctors::forward_change<float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner,
    Tensor2<float> &embedding_feature, cudaStream_t stream, float desired_value);

template void SparseEmbeddingFunctors::forward_change<__half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner,
    Tensor2<__half> &embedding_feature, cudaStream_t stream, float desired_value);

template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::forward_sum_change_(size_t batch_size, size_t slot_num,
                                                  size_t embedding_vec_size, int combiner,
                                                  Tensor2<TypeEmbeddingComp> &embedding_feature,
                                                  cudaStream_t stream, float desired_value) {
  TypeEmbeddingComp *temp_device_ptr = nullptr;
  size_t num_elements = embedding_feature.get_num_elements();
  cudaMalloc(&temp_device_ptr, num_elements * sizeof(TypeEmbeddingComp));
  cudaMemcpyAsync(temp_device_ptr, embedding_feature.get_ptr(),
                  num_elements * sizeof(TypeEmbeddingComp), cudaMemcpyDeviceToDevice);
  try {
    // do sum reduction
    if (combiner == 0) {
      forward_sum_change(batch_size, slot_num, embedding_vec_size, temp_device_ptr, stream,
                         desired_value);
    } else if (combiner == 1) {
      forward_mean_change(batch_size, slot_num, embedding_vec_size, temp_device_ptr, stream,
                          desired_value);
    } else {
      HCTR_OWN_THROW(Error_t::WrongInput, "Invalid combiner type ");
    }
  } catch (const std::runtime_error &rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }

  cudaMemcpyAsync(embedding_feature.get_ptr(), temp_device_ptr,
                  num_elements * sizeof(TypeEmbeddingComp), cudaMemcpyDeviceToDevice);

  cudaDeviceSynchronize();
  cudaFree(temp_device_ptr);
  return;
}

template void SparseEmbeddingFunctors::forward_sum_change_<float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner,
    Tensor2<float> &embedding_feature, cudaStream_t stream, float desired_value);

template void SparseEmbeddingFunctors::forward_sum_change_<__half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner,
    Tensor2<__half> &embedding_feature, cudaStream_t stream, float desired_value);
}  // namespace HugeCTR
