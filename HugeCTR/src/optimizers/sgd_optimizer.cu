/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "HugeCTR/include/optimizers/sgd_optimizer.hpp"

namespace {

__global__ void sgd_kernel(int len, float* weight, const float* wgrad, float lr, float scaler) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float gi = wgrad[i] / scaler;
    weight[i] -= lr * gi;
  }
}

}  // namespace

namespace HugeCTR {

void SgdOptimizer::update(cudaStream_t stream) {
  CudaDeviceContext context(device_id_);

  const int len = weight_->get_num_elements();
  const int block_dim = 256;
  const int grid_dim = (len - 1) / block_dim + 1;

  float* weight = weight_->get_ptr_with_offset(0);
  const float* wgrad = wgrad_->get_ptr_with_offset(0);

  sgd_kernel<<<grid_dim, block_dim, 0, stream>>>(len, weight, wgrad, lr_, scaler_);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

}  // namespace HugeCTR
