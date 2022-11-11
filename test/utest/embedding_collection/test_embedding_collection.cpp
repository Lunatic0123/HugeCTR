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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/time.h>

#include <numeric>

#include "HugeCTR/core/hctr_impl/hctr_backend.hpp"
#include "HugeCTR/embedding/embedding.hpp"
#include "HugeCTR/include/embeddings/embedding_collection.hpp"
#include "HugeCTR/include/resource_managers/resource_manager_ext.hpp"
#include "embedding_collection_cpu.hpp"
#include "embedding_collection_utils.hpp"
#include "embeddings/embedding_collection.hpp"
using namespace embedding;

const int batch_size = 8192;
// table params
const int num_table = 4;
const std::vector<int> table_ev_size_list = {128, 64, 32, 16};
const std::vector<int> table_max_vocabulary_list = {398844, 39043, 17289, 124345};

// lookup params
const std::vector<LookupParam> lookup_params = {
    {0, 0, Combiner::Sum, 8, table_ev_size_list[0]},
    {1, 1, Combiner::Average, 20, table_ev_size_list[1]},
    {2, 2, Combiner::Sum, 10, table_ev_size_list[2]},
    {3, 3, Combiner::Average, 5, table_ev_size_list[3]},
};

const std::vector<LookupParam> lookup_params_with_shared_table = {
    {0, 0, Combiner::Sum, 8, table_ev_size_list[0]},
    {1, 1, Combiner::Average, 20, table_ev_size_list[1]},
    {2, 2, Combiner::Sum, 10, table_ev_size_list[2]},
    {3, 3, Combiner::Average, 5, table_ev_size_list[3]},
};

const std::vector<int> device_list = {0, 1};
bool debug_verbose = false;

std::vector<EmbeddingTableParam> get_table_param_list(core::DataType emb_type) {
  std::vector<EmbeddingTableParam> table_param_list;

  HugeCTR::OptParams opt_param;
  // FIXME: We need to initialize all variable or we will trigger uninitialized error in
  // EmbeddingTableParam ctor because the copy constructor of HugeCTR::OptParams trys to copy all
  // members
  opt_param.optimizer = HugeCTR::Optimizer_t::SGD;
  opt_param.lr = 1e-1;
  opt_param.scaler = (emb_type == TensorScalarType::Float16) ? 1024 : 1;
  opt_param.hyperparams = HugeCTR::OptHyperParams{};
  opt_param.update_type = HugeCTR::Update_t::Local;

  InitParams init_param;
  for (int table_id = 0; table_id < num_table; ++table_id) {
    EmbeddingTableParam table_param{table_id, table_max_vocabulary_list[table_id],
                                    table_ev_size_list[table_id], opt_param, init_param};
    table_param_list.push_back(std::move(table_param));
  }
  return table_param_list;
}

template <typename key_t, typename offset_t, typename index_t, typename emb_t>
void embedding_collection_e2e(const std::vector<LookupParam> &lookup_params,
                              const std::vector<std::vector<int>> &shard_matrix,
                              const std::vector<GroupedEmbeddingParam> &grouped_emb_params) {
  ASSERT_EQ(table_max_vocabulary_list.size(), num_table);
  ASSERT_EQ(table_ev_size_list.size(), num_table);

  EmbeddingCollectionParam ebc_param{num_table,
                                     static_cast<int>(lookup_params.size()),
                                     lookup_params,
                                     shard_matrix,
                                     grouped_emb_params,
                                     batch_size,
                                     HugeCTR::TensorScalarTypeFunc<key_t>::get_type(),
                                     HugeCTR::TensorScalarTypeFunc<index_t>::get_type(),
                                     HugeCTR::TensorScalarTypeFunc<offset_t>::get_type(),
                                     HugeCTR::TensorScalarTypeFunc<emb_t>::get_type(),
                                     EmbeddingLayout::FeatureMajor};
  auto table_param_list = get_table_param_list(ebc_param.emb_type);

  auto resource_manager = HugeCTR::ResourceManagerExt::create({device_list}, 0);
  int num_gpus = static_cast<int>(device_list.size());

  std::vector<key_t> key_list;
  std::vector<offset_t> bucket_range;
  auto prepare_input = [&] {
    timeval t1;
    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);
    key_list.clear();
    bucket_range.clear();
    bucket_range.push_back(0);

    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      auto &lookup_param = ebc_param.lookup_params[lookup_id];
      int table_id = lookup_param.table_id;
      int max_hotness = lookup_param.max_hotness;
      auto &table_param = table_param_list[table_id];

      for (int b = 0; b < ebc_param.universal_batch_size; ++b) {
        int nnz = (lookup_param.combiner == Combiner::Concat)
                      ? max_hotness
                      : 1 + rand() % max_hotness;  // TODO: support nnz=0
        bucket_range.push_back(nnz);
        for (int i = 0; i < nnz; ++i) {
          key_t key = rand() % table_param.max_vocabulary_size;
          key_list.push_back(key);
        }
      }
    }
    std::inclusive_scan(bucket_range.begin(), bucket_range.end(), bucket_range.begin());
  };

  std::vector<std::vector<emb_t>> top_grads;
  auto prepare_top_grads = [&] {
    top_grads.resize(num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      top_grads[gpu_id].clear();
      for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
        auto &lookup_param = ebc_param.lookup_params[lookup_id];
        int num_ev = (lookup_param.combiner == Combiner::Concat) ? lookup_param.max_hotness : 1;
        for (int b = 0;
             b < ebc_param.universal_batch_size * lookup_param.ev_size * num_ev / num_gpus; ++b) {
          float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
          top_grads[gpu_id].push_back(HugeCTR::TypeConvert<emb_t, float>::convert(r));
        }
      }
    }
  };

  std::vector<std::shared_ptr<core::CoreResourceManager>> core_resource_manager_list;

  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, gpu_id);

    core_resource_manager_list.push_back(core);
  }

  std::unique_ptr<embedding::EmbeddingCollection> ebc =
      std::make_unique<embedding::EmbeddingCollection>(resource_manager, core_resource_manager_list,
                                                       ebc_param, ebc_param, table_param_list);

  std::vector<core::Tensor> ebc_key_list;
  std::vector<core::Tensor> ebc_bucket_range_list;
  std::vector<size_t *> ebc_num_keys_list;
  std::vector<core::Tensor> ebc_top_grads;
  std::vector<core::Tensor> ebc_outptut;
  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());
    auto buffer = GetBuffer(core_resource_manager_list[gpu_id]);

    int max_hotness_sum = 0;
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      auto &lookup_param = ebc_param.lookup_params[lookup_id];
      int max_hotness = lookup_param.max_hotness;
      max_hotness_sum += max_hotness;
    }

    ebc_key_list.push_back(buffer->reserve({ebc_param.universal_batch_size, max_hotness_sum},
                                           DeviceType::GPU, ebc_param.key_type));
    ebc_bucket_range_list.push_back(
        buffer->reserve({ebc_param.universal_batch_size * ebc_param.num_lookup + 1},
                        DeviceType::GPU, ebc_param.offset_type));
    ebc_num_keys_list.push_back(new size_t);

    int64_t num_ev = 0;
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      auto &lookup_param = ebc_param.lookup_params[lookup_id];
      num_ev += (lookup_param.combiner == Combiner::Concat)
                    ? lookup_param.ev_size * lookup_param.max_hotness
                    : lookup_param.ev_size;
    }
    num_ev *= (ebc_param.universal_batch_size / num_gpus);
    ebc_top_grads.push_back(buffer->reserve(num_ev, DeviceType::GPU, ebc_param.emb_type));
    ebc_outptut.push_back(buffer->reserve(num_ev, DeviceType::GPU, ebc_param.emb_type));
    buffer->allocate();
  }

  auto prepare_gpu_input = [&] {
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());

      ebc_key_list[gpu_id].copy_from(key_list);
      ebc_bucket_range_list[gpu_id].copy_from(bucket_range);
      *(ebc_num_keys_list[gpu_id]) = key_list.size();
      ebc_top_grads[gpu_id].copy_from(top_grads[gpu_id]);
    }
  };

  auto prepare_data = [&] {
    prepare_input();
    prepare_top_grads();
    prepare_gpu_input();
  };

  auto sync_gpus = [&]() {
    for (auto core : core_resource_manager_list) {
      HCTR_LIB_THROW(cudaStreamSynchronize(core->get_local_gpu()->get_stream()));
    }
  };
  // sync for emb table init
  sync_gpus();

  std::vector<std::vector<IGroupedEmbeddingTable *>> grouped_emb_table_ptr_list =
      ebc->get_grouped_embedding_tables();

  EmbeddingCollectionCPU<key_t, offset_t, index_t, emb_t> ebc_cpu{
      num_gpus, ebc_param, num_table, table_param_list, grouped_emb_table_ptr_list};

  EmbeddingReferenceCPU<key_t, offset_t, index_t, emb_t> emb_ref{
      num_gpus, ebc_param, num_table, table_param_list, grouped_emb_table_ptr_list};

  auto check_forward_result = [&] {
    std::cout << "compare ebc cpu emb output vs. emb reference emb output.\n";
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      ASSERT_EQ(ebc_cpu.embedding_vec_[gpu_id].size(), emb_ref.embedding_vec_[gpu_id].size());
      // std::cout << "forward cpu output:\n";
      // print_array(ebc_cpu.embedding_vec_[gpu_id].size(),
      // ebc_cpu.embedding_vec_[gpu_id]); std::cout << "forward ref output:\n";
      // print_array(emb_ref.embedding_vec_[gpu_id].size(),
      // emb_ref.embedding_vec_[gpu_id]);
      assert_array_eq(ebc_cpu.embedding_vec_[gpu_id].size(), ebc_cpu.embedding_vec_[gpu_id],
                      emb_ref.embedding_vec_[gpu_id]);
    }
    std::cout << "\t>pass compare ebc cpu emb output vs. emb reference emb output.\n";

    std::cout << "compare ebc gpu emb output vs. emb reference emb output.\n";
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      std::vector<emb_t> gpu_emb_output;
      ebc_outptut[gpu_id].to(&gpu_emb_output);
      ASSERT_EQ(gpu_emb_output.size(), emb_ref.embedding_vec_[gpu_id].size());
      if (debug_verbose) {
        std::cout << "forward ref output:\n";
        print_array(emb_ref.embedding_vec_[gpu_id].size(), emb_ref.embedding_vec_[gpu_id]);
        std::cout << "forward gpu output:\n";
        print_array(gpu_emb_output.size(), gpu_emb_output);
      }
      assert_array_eq(gpu_emb_output.size(), gpu_emb_output, ebc_cpu.embedding_vec_[gpu_id]);
    }
    std::cout << "\t>pass compare ebc gpu emb output vs. emb reference emb output.\n";
  };
  auto check_backward_result = [&] {
    auto compare_grad_in_table = [](const std::unordered_map<key_t, std::vector<float>> &lhs,
                                    const std::unordered_map<key_t, std::vector<float>> &rhs) {
      ASSERT_EQ(lhs.size(), rhs.size());

      for (auto p : lhs) {
        auto &k = p.first;
        auto &lhs_ev = p.second;
        ASSERT_TRUE(rhs.find(k) != rhs.end());
        auto &rhs_ev = rhs.at(k);
        ASSERT_EQ(lhs_ev.size(), rhs_ev.size());
        // if (debug_verbose) {
        //   std::cout << "lhs output:\n";
        //   print_array(lhs_ev.size(), lhs_ev);
        //   std::cout << "rhs output:\n";
        //   print_array(rhs_ev.size(), rhs_ev);
        // }
        assert_array_eq(lhs_ev.size(), lhs_ev, rhs_ev);
      }
    };

    std::cout << "compare ref grad info vs. ebc cpu grad info.\n";
    ASSERT_EQ(ebc_cpu.grad_info_.size(), emb_ref.accumulate_grad_map_.size());
    for (int table_id = 0; table_id < num_table; ++table_id) {
      ASSERT_TRUE(table_id < static_cast<int>(ebc_cpu.grad_info_.size()));
      auto &cpu_grad_in_table = ebc_cpu.grad_info_.at(table_id);
      auto &ref_grad_in_table = emb_ref.accumulate_grad_map_.at(table_id);
      compare_grad_in_table(cpu_grad_in_table, ref_grad_in_table);
    }
    std::cout << "\t>pass compare ref grad info vs. ebc cpu grad info.\n";
  };

  auto check_embedding_table = [&] {
    std::cout << "compare ref emb table vs. ebc cpu emb table.\n";
    const auto &cpu_emb_table = ebc_cpu.emb_table_cpu_.emb_table_list_;
    const auto &ref_emb_table = emb_ref.emb_table_cpu_.emb_table_list_;
    ASSERT_TRUE(cpu_emb_table.size() == ref_emb_table.size());

    for (size_t table_id = 0; table_id < cpu_emb_table.size(); ++table_id) {
      ASSERT_EQ(cpu_emb_table[table_id].size(), ref_emb_table[table_id].size());

      for (auto &[k, cpu_ev] : cpu_emb_table[table_id]) {
        ASSERT_TRUE(cpu_emb_table[table_id].find(k) != ref_emb_table[table_id].end());
        auto ref_ev = ref_emb_table[table_id].at(k);

        ASSERT_EQ(cpu_ev.size(), ref_ev.size());
        assert_array_eq(cpu_ev.size(), cpu_ev, ref_ev);
      }
    }
    std::cout << "\t>pass compare ref emb table vs. ebc cpu emb table.\n";

    // EmbeddingTableCPU<key_t, index_t> copy_gpu_emb_table{num_table,
    // table_major_ebc_table_ptr_list,
    //                                                      table_param_list};
    // const auto &gpu_emb_table = copy_gpu_emb_table.emb_table_list_;

    // std::cout << "compare ref emb table vs. ebc gpu emb table.\n";
    // ASSERT_TRUE(gpu_emb_table.size() == ref_emb_table.size());

    // for (size_t id_space = 0; id_space < gpu_emb_table.size(); ++id_space) {
    //   ASSERT_EQ(gpu_emb_table[id_space].size(),
    //   ref_emb_table[id_space].size());

    //   for (auto &[k, gpu_ev] : gpu_emb_table[id_space]) {
    //     ASSERT_TRUE(gpu_emb_table[id_space].find(k) !=
    //     ref_emb_table[id_space].end()); auto ref_ev =
    //     ref_emb_table[id_space].at(k);

    //     ASSERT_EQ(gpu_ev.size(), ref_ev.size());
    //     assert_array_eq(gpu_ev.size(), gpu_ev, ref_ev);
    //   }
    // }
    // std::cout << "\t>pass compare ref emb table vs. ebc gpu emb table.\n";
  };

  int num_iteration = 10;
  for (int iter = 0; iter < num_iteration; ++iter) {
    std::cout << "iter:" << iter << "\n";
    prepare_data();
    sync_gpus();

    // forward
    ebc_cpu.embedding_forward_cpu(key_list, bucket_range);
    emb_ref.embedding_forward_cpu(key_list, bucket_range);
#pragma omp parallel for num_threads(num_gpus)
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      ebc->forward_per_gpu(true, gpu_id, ebc_key_list[gpu_id], ebc_bucket_range_list[gpu_id],
                           *ebc_num_keys_list[gpu_id], ebc_outptut[gpu_id]);
    }
    sync_gpus();
    check_forward_result();

    // backward
    ebc_cpu.embedding_backward_cpu(top_grads, batch_size);
    emb_ref.embedding_backward_cpu(top_grads, key_list, bucket_range);
#pragma omp parallel for num_threads(num_gpus)
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      ebc->backward_per_gpu(gpu_id, ebc_top_grads[gpu_id], true);
    }
    sync_gpus();
    check_backward_result();

    // update
    ebc_cpu.embedding_update_cpu();
    emb_ref.embedding_update_cpu();
#pragma omp parallel for num_threads(num_gpus)
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      ebc->update_per_gpu(gpu_id);
    }
    sync_gpus();

    check_embedding_table();
  }
}

// dp
namespace dp {
const std::vector<std::vector<int>> shard_matrix = {
    {1, 1, 1, 1},
    {1, 1, 1, 1},
};

const std::vector<GroupedEmbeddingParam> grouped_emb_params = {
    {TablePlacementStrategy::DataParallel, {0, 1, 2, 3}}};

TEST(test_embedding_collection, dp_plan0) {
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(lookup_params, shard_matrix,
                                                                grouped_emb_params);
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, __half>(lookup_params, shard_matrix,
                                                                 grouped_emb_params);
}
}  // namespace dp

namespace mp {
const std::vector<std::vector<int>> shard_matrix = {
    {1, 0, 1, 1},
    {0, 1, 1, 1},
};

const std::vector<GroupedEmbeddingParam> grouped_emb_params = {
    {TablePlacementStrategy::ModelParallel, {0, 1, 2, 3}}};

TEST(test_embedding_collection, mp_plan0) {
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(lookup_params, shard_matrix,
                                                                grouped_emb_params);
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, __half>(lookup_params, shard_matrix,
                                                                 grouped_emb_params);
}

TEST(test_embedding_collection, mp_plan1) {
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(lookup_params_with_shared_table,
                                                                shard_matrix, grouped_emb_params);
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, __half>(lookup_params_with_shared_table,
                                                                 shard_matrix, grouped_emb_params);
}
}  // namespace mp

namespace dp_and_mp {
const std::vector<std::vector<int>> shard_matrix = {
    {1, 0, 1, 1},
    {0, 1, 1, 1},
};

const std::vector<GroupedEmbeddingParam> grouped_emb_params = {
    {TablePlacementStrategy::DataParallel, {2}},
    {TablePlacementStrategy::ModelParallel, {0, 1, 3}}};

TEST(test_embedding_collection, dp_and_mp_plan0) {
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(lookup_params, shard_matrix,
                                                                grouped_emb_params);
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, __half>(lookup_params, shard_matrix,
                                                                 grouped_emb_params);
}

TEST(test_embedding_collection, dp_and_mp_plan1) {
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(lookup_params_with_shared_table,
                                                                shard_matrix, grouped_emb_params);
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, __half>(lookup_params_with_shared_table,
                                                                 shard_matrix, grouped_emb_params);
}
}  // namespace dp_and_mp