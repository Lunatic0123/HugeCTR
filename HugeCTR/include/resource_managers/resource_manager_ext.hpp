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
#include <resource_manager.hpp>
#include <resource_managers/resource_manager_core.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace HugeCTR {

/**
 * @brief GPU resources manager which holds all the resources required by training
 *
 * An extended GPU Resource manager
 */
class ResourceManagerExt : public ResourceManager {
  std::shared_ptr<ResourceManager> core_;
  std::vector<std::shared_ptr<rmm::mr::device_memory_resource>> base_cuda_mr_;
  std::vector<std::shared_ptr<rmm::mr::device_memory_resource>> memory_resource_;

  void initialize_rmm_resources();

  ResourceManagerExt(std::shared_ptr<ResourceManager> core);
 public:
  static std::shared_ptr<ResourceManager> create(
      const std::vector<std::vector<int>>& visible_devices, unsigned long long seed);

  ResourceManagerExt(const ResourceManagerExt&) = delete;
  ResourceManagerExt& operator=(const ResourceManagerExt&) = delete;

  // from ResourceManagerBase
  void set_local_gpu(std::shared_ptr<GPUResource> gpu_resource, size_t local_gpu_id) override { core_->set_local_gpu(gpu_resource, local_gpu_id);
  }
  const std::shared_ptr<GPUResource>& get_local_gpu(size_t local_gpu_id) const override {
    return core_->get_local_gpu(local_gpu_id);
  }
  size_t get_local_gpu_count() const override { return core_->get_local_gpu_count(); }
  size_t get_global_gpu_count() const override { return core_->get_global_gpu_count(); }


  // from ResourceManager
  int get_num_process() const override { return core_->get_num_process(); }
  int get_process_id() const override { return core_->get_process_id(); }
  int get_master_process_id() const override { return core_->get_master_process_id(); }
  bool is_master_process() const override { return core_->is_master_process(); }

  const std::shared_ptr<CPUResource>& get_local_cpu() const override {
    return core_->get_local_cpu();
  }

  const std::vector<int>& get_local_gpu_device_id_list() const override {
    return core_->get_local_gpu_device_id_list();
  }

  int get_process_id_from_gpu_global_id(size_t global_gpu_id) const override {
    return core_->get_process_id_from_gpu_global_id(global_gpu_id);
  }

  size_t get_gpu_local_id_from_global_id(size_t global_gpu_id) const override {
    return core_->get_gpu_local_id_from_global_id(global_gpu_id);
  }

  size_t get_gpu_global_id_from_local_id(size_t local_gpu_id) const override {
    return core_->get_gpu_global_id_from_local_id(local_gpu_id);
  }

  bool p2p_enabled(int src_dev, int dst_dev) const override {
    return core_->p2p_enabled(src_dev, dst_dev);
  }
  bool all_p2p_enabled() const override {
    return core_->all_p2p_enabled();
  }

  // its own methods
  const std::shared_ptr<rmm::mr::device_memory_resource>& get_device_rmm_device_memory_resource(
      int local_gpu_id) const;
};
}  // namespace HugeCTR
