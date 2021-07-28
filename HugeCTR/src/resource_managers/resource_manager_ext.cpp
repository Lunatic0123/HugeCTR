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

#include <resource_managers/resource_manager_ext.hpp>
#include <random>
#include <utils.hpp>

namespace HugeCTR {

std::unordered_map<int, int> CudaCPUDeviceContext::device_id_to_numa_node_;

std::shared_ptr<ResourceManager> ResourceManagerExt::create(
    const std::vector<std::vector<int>>& visible_devices, unsigned long long seed, DeviceMap::Layout layout) {
  int size = 1, rank = 0;

#ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &size));
  CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
#endif

  DeviceMap device_map(visible_devices, rank, layout);

  std::random_device rd;
  if (seed == 0) {
    seed = rd();
  }

#ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD));
#endif

  MESSAGE_("Global seed is " + std::to_string(seed));
  
  CK_NVML_THROW_(nvmlInit_v2());
  CudaCPUDeviceContext::init_cpu_mapping(device_map.get_device_list());

  std::shared_ptr<ResourceManager> core(
      new ResourceManagerCore(size, rank, std::move(device_map), seed));

  return std::shared_ptr<ResourceManager>(new ResourceManagerExt(core));
}

ResourceManagerExt::ResourceManagerExt(std::shared_ptr<ResourceManager> core)
    : core_(core) {
#ifdef ENABLE_MPI
  int num_process = get_num_process();
  if (num_process > 1) {
    int process_id = get_process_id();
    ib_comm_ = std::make_unique<IbComm>();
    ib_comm_->init(num_process, get_local_gpu_count(), process_id, get_local_gpu_device_id_list());
  }
#endif
}

void ResourceManagerExt::set_ar_comm(AllReduceAlgo algo, bool use_mixed_precision)
{
  int num_process = get_num_process();
#ifdef ENABLE_MPI
  ar_comm_ = AllReduceInPlaceComm::create(num_process, algo, use_mixed_precision, get_local_gpus(),
                                          ib_comm_.get());
#else
  ar_comm_ = AllReduceInPlaceComm::create(num_process, algo, use_mixed_precision, get_local_gpus());
#endif
}

}  // namespace HugeCTR
