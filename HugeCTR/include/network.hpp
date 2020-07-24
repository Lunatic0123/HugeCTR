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

#pragma once

#include <cublas_v2.h>
#include <nccl.h>
#include <fstream>
#include <functional>
#include <vector>
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/layer.hpp"
#include "HugeCTR/include/loss.hpp"
#include "HugeCTR/include/metrics.hpp"
#include "HugeCTR/include/optimizer.hpp"
#include "HugeCTR/include/tensor.hpp"
#include "nlohmann/json.hpp"

namespace HugeCTR {

/**
 * @brief Dense network (embedding is not included)
 *
 * Each GPU (device) has an instance of Network. Network performs
 * forward/backward/loss/update of the dense layers.
 */
class Network {
  friend Network* create_network(
      const nlohmann::json& j_array, const nlohmann::json& j_optimizor,
      const std::map<std::string, std::shared_ptr<ITensor>>& tensor_list_in, int device_id,
      int num_networks_in_global, const std::shared_ptr<const GPUResource>& gpu_resource,
      bool use_mixed_precision, float scaler,
      bool use_algorithm_search);

 private:
  //  Tensors<float> tensors_;
  std::vector<std::unique_ptr<Layer>> layers_;              /**< vector of layers */
  std::shared_ptr<GeneralBuffer<float>> blobs_buff_;        /**< blobs' general buffer */
  std::shared_ptr<GeneralBuffer<float>> weight_buff_;       /**< weight (param) general buffer */
  std::shared_ptr<GeneralBuffer<float>> wgrad_buff_;        /**< weight gradient general buffer */

  std::shared_ptr<GeneralBuffer<__half>> blobs_buff_half_;  /**< blobs' general buffer */
  std::shared_ptr<GeneralBuffer<__half>> weight_buff_half_; /**< weight (param) general buffer */
  std::shared_ptr<GeneralBuffer<__half>> wgrad_buff_half_;  /**< weight gradient general buffer */

  std::shared_ptr<const GPUResource> gpu_resource_;         /**< gpu resource */
  int device_id_;                                           /**< device id */
  std::unique_ptr<Optimizer> optimizer_;                    /**< optimizer */
  std::unique_ptr<ILoss> loss_;                             /**< loss */
  std::shared_ptr<Tensor<float>> loss_tensor_;              /**< loss tensor */
  bool full_fp16_{false};
  bool enable_cuda_graph_{true};
  int n_sms_{0};
  void conv_weight_();
  metrics::RawMetricMap raw_metrics_;

  bool eval_graph_created_;
  bool train_fprop_graph_created_;
  bool train_bprop_graph_created_;
  cudaGraph_t eval_graph_;
  cudaGraph_t train_fprop_graph_;
  cudaGraph_t train_bprop_graph_;
  cudaGraphExec_t eval_instance_;
  cudaGraphExec_t train_fprop_instance_;
  cudaGraphExec_t train_bprop_instance_;

 public:
  /**
   * Ctor.
   * @param device_id device id.
   * @param gpu_resource gpu resource for local gpu.
   * @param disable_parser only for unit test.
   */
  Network(int device_id, const std::shared_ptr<const GPUResource>& gpu_resource,
          bool full_fp16 = false);
  Network(const Network& C) = delete;
  Network& operator=(const Network&) = delete;

  std::shared_ptr<GeneralBuffer<float>>& get_weight() { return weight_buff_; }

  std::shared_ptr<GeneralBuffer<__half>>& get_weight_half() { return weight_buff_half_; }

  void set_weight(std::shared_ptr<GeneralBuffer<float>>& weight_buff) {
    weight_buff_->replace_buffer_with(*weight_buff);
    return;
  }

  void set_weight_half(std::shared_ptr<GeneralBuffer<__half>>& wgrad_buff_half) {
    weight_buff_half_->replace_buffer_with(*wgrad_buff_half);
    return;
  }

  /**
   * Forward, backward and update the network.
   */
  void train();

  /**
   * Forward only.
   */
  void eval();

  /**
   * Get current loss and return.
   */
  float get_loss();

  metrics::RawMetricMap get_raw_metrics() const;

  /**
   * Get number of parameters in this network.
   */
  size_t get_params_num() const { return weight_buff_->get_num_elements(); }

  /**
   * Writting paramters to fstream.
   */
  void download_params_to_host(std::ofstream& weight_stream);

  /**
   * Get no trained parameters (such as parameters in Batch nomalization) to string.
   */
  std::string get_no_trained_params_in_string();

  /**
   * Read parameters from model_file.
   */
  void upload_params_to_device(const std::string& model_file);

  /**
   * Writting paramters to cpu buffer.
   */
  void download_params_to_host(float* weight);

  /**
   * Read parameters from cpu buffer.
   */
  void upload_params_to_device(float* params);

  /**
   * Init parameters and write to fstream.
   */
  void init_params(const std::string& model_file_name);

  /**
   * Copy parameters from a network.
   */
  void copy_params(const Network& n);

  /**
   * Exchange wgrad between gpus.
   */
  void exchange_wgrad();

  /**
   * Update parameters.
   */
  void update_params();

  /**
   * reset the learning rate to lr.
   */
  void set_learning_rate(float lr) { optimizer_->set_learning_rate(lr); }

  /**
   * initialize layer by layer
   */
  void initialize() {
    for (auto& n : layers_) {
      n->initialize();
    }
  }

  /**
   * search_algorithm layer by layer
   */
  void search_algorithm() {
    for (auto& n : layers_) {
      n->search_algorithm();
    }
  }
};

}  // namespace HugeCTR
