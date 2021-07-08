/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "HugeCTR/include/layers/elu_layer.hpp"

#include <math.h>

#include <vector>

#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace std;
using namespace HugeCTR;

namespace {

template <typename T>
struct Eps {
  static T value();
};

template <>
struct Eps<float> {
  static constexpr float value() { return 1e-6f; }
};

template <>
struct Eps<__half> {
  static __half value() { return __float2half(1e-2f); }
};

template <typename T>
void elu_cpu(const T* in, T* out, int len, T alpha) {
  for (int i = 0; i < len; ++i) {
    out[i] = (in[i] < 0) ? T(alpha * (exp(in[i]) - 1)) : in[i];
  }
}

template <typename T>
void elu_bprop_cpu(const T* d_out, T* d_in, int len, T alpha) {
  for (int i = 0; i < len; ++i) {
    d_in[i] = (d_in[i] < 0) ? T(alpha * exp(d_in[i]) * d_out[i]) : d_out[i];
  }
}

template <typename T>
void elu_test(size_t dim0, size_t dim1, T alpha) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
  vector<size_t> dims = {dim0, dim1};

  Tensor2<T> in_tensor;
  buf->reserve(dims, &in_tensor);
  Tensor2<T> out_tensor;
  buf->reserve(dims, &out_tensor);

  EluLayer<T> elu_layer(in_tensor, out_tensor, alpha, test::get_default_gpu());

  buf->allocate();
  elu_layer.initialize();

  const int len = dim0 * dim1;
  T* d_in = in_tensor.get_ptr();
  T* d_out = out_tensor.get_ptr();
  std::unique_ptr<T[]> h_in(new T[len]);
  std::unique_ptr<T[]> h_out(new T[len]);
  std::unique_ptr<T[]> h_expected(new T[len]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  // fprop
  simulator.fill(h_in.get(), len);
  CK_CUDA_THROW_(cudaMemcpy(d_in, h_in.get(), len * sizeof(T), cudaMemcpyHostToDevice));

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  elu_layer.fprop(true);
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  CK_CUDA_THROW_(cudaMemcpy(h_out.get(), d_out, len * sizeof(T), cudaMemcpyDeviceToHost));

  elu_cpu(h_in.get(), h_expected.get(), len, alpha);
  ASSERT_TRUE(test::compare_array_approx<T>(h_out.get(), h_expected.get(), len, Eps<T>::value()));

  // bprop
  simulator.fill(h_in.get(), len);
  simulator.fill(h_out.get(), len);
  for (int i = 0; i < len; ++i) {
    h_expected[i] = h_in[i];
  }
  CK_CUDA_THROW_(cudaMemcpy(d_in, h_in.get(), len * sizeof(T), cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(cudaMemcpy(d_out, h_out.get(), len * sizeof(T), cudaMemcpyHostToDevice));

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  elu_layer.bprop();
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  CK_CUDA_THROW_(cudaMemcpy(h_in.get(), d_in, len * sizeof(T), cudaMemcpyDeviceToHost));

  elu_bprop_cpu(h_out.get(), h_expected.get(), len, alpha);
  ASSERT_TRUE(test::compare_array_approx<T>(h_in.get(), h_expected.get(), len, Eps<T>::value()));
}

}  // namespace

TEST(elu_layer, fp32_10x20_1) { elu_test<float>(10, 20, 1.0); }
TEST(elu_layer, fp32_10x500_1) { elu_test<float>(10, 500, 1.0); }
TEST(elu_layer, fp32_512x2048_1) { elu_test<float>(512, 1024 * 2, 1.0); }
TEST(elu_layer, fp16_10x20_1) { elu_test<__half>(10, 20, 1.0); }
TEST(elu_layer, fp16_10x500_1) { elu_test<__half>(10, 500, 1.0); }
TEST(elu_layer, fp16_512x2048_1) { elu_test<__half>(512, 1024 * 2, 1.0); }
