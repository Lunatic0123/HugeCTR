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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/time.h>

#include <core/hctr_impl/hctr_backend.hpp>
#include <embedding/embedding.hpp>
#include <embeddings/embedding_collection.hpp>
#include <numeric>
#include <resource_managers/resource_manager_ext.hpp>
#include <utest/embedding_collection/embedding_collection_utils.hpp>
#include <utest/embedding_collection/reference_embedding.hpp>

using namespace embedding;

const int batch_size = 8192;
const int num_iteration = 10;
const bool debug_verbose = false;
// table params
const int num_table = 4;
const std::vector<int> table_ev_size_list = {128, 64, 32, 16};
const std::vector<int> table_max_vocabulary_list = {398844, 39043, 17289, 124345};

// lookup params
const std::vector<LookupParam> lookup_params0 = {
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
    {4, 1, Combiner::Average, 8, table_ev_size_list[1]},
};

const std::vector<int> device_list = {0, 1, 2, 3, 4, 5, 6, 7};

std::vector<EmbeddingTableParam> get_table_param_list(core23::DataType emb_type) {
  std::vector<EmbeddingTableParam> table_param_list;

  HugeCTR::OptParams opt_param;
  // FIXME: We need to initialize all variable or we will trigger uninitialized error in
  // EmbeddingTableParam ctor because the copy constructor of HugeCTR::OptParams tries to copy all
  // members
  opt_param.optimizer = HugeCTR::Optimizer_t::SGD;
  opt_param.lr = 1e-1;
  opt_param.scaler = (emb_type == core23::ScalarType::Half) ? 1024 : 1;
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
                              const std::vector<GroupedTableParam> &grouped_emb_params,
                              EmbeddingLayout output_layout, SortStrategy sort_strategy,
                              AllreduceStrategy allreduce_strategy) {
  ASSERT_EQ(table_max_vocabulary_list.size(), num_table);
  ASSERT_EQ(table_ev_size_list.size(), num_table);
  auto key_type = HugeCTR::core23::ToScalarType<key_t>::value;
  auto index_type = HugeCTR::core23::ToScalarType<index_t>::value;
  auto offset_type = HugeCTR::core23::ToScalarType<offset_t>::value;
  auto emb_type = HugeCTR::core23::ToScalarType<emb_t>::value;
  auto wgrad_type = HugeCTR::core23::ToScalarType<emb_t>::value;

  std::cout << "embedding_collection_e2e test. output_layout:" << output_layout
            << ", sort_strategy:" << sort_strategy << ", allreduce_strategy:" << allreduce_strategy
            << ", key_type:" << key_type << ", emb_type:" << emb_type << "\n";
  EmbeddingCollectionParam ebc_param{num_table,
                                     static_cast<int>(lookup_params.size()),
                                     lookup_params,
                                     shard_matrix,
                                     grouped_emb_params,
                                     batch_size,
                                     key_type,
                                     index_type,
                                     offset_type,
                                     emb_type,
                                     wgrad_type,
                                     EmbeddingLayout::FeatureMajor,
                                     output_layout,
                                     sort_strategy,
                                     embedding::KeysPreprocessStrategy::AddOffset,
                                     allreduce_strategy,
                                     CommunicationStrategy::Uniform};
  auto table_param_list = get_table_param_list(ebc_param.emb_type);

  auto resource_manager = HugeCTR::ResourceManagerExt::create({device_list}, 0);
  int num_gpus = static_cast<int>(device_list.size());
  int batch_size_per_gpu = batch_size / num_gpus;

  std::vector<key_t> key_list;
  std::vector<offset_t> bucket_range;
  std::vector<std::vector<std::vector<key_t>>> dp_keys;
  std::vector<std::vector<std::vector<offset_t>>> dp_num_keys_per_bucket;
  auto prepare_input = [&] {
    timeval t1;
    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);
    key_list.clear();
    bucket_range.clear();
    dp_keys.clear();
    dp_num_keys_per_bucket.clear();

    bucket_range.push_back(0);
    dp_keys.resize(num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      dp_keys[gpu_id].resize(ebc_param.num_lookup);
    }
    dp_num_keys_per_bucket.resize(num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      dp_num_keys_per_bucket[gpu_id].resize(ebc_param.num_lookup);
    }

    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      auto &lookup_param = ebc_param.lookup_params[lookup_id];
      int max_hotness = lookup_param.max_hotness;

      std::vector<std::vector<key_t>> dp_keys_on_one_gpu;
      for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        for (int b = 0; b < batch_size_per_gpu; ++b) {
          int nnz = max_hotness;  // FIXME: static nnz
          /*
              int nnz = (lookup_param.combiner == Combiner::Concat)
                            ? max_hotness
                            : 1 + rand() % max_hotness;  // TODO: support nnz=0
          */

          dp_num_keys_per_bucket[gpu_id][lookup_id].push_back(nnz);

          bucket_range.push_back(nnz);
          for (int i = 0; i < nnz; ++i) {
            int table_id = lookup_param.table_id;
            auto &table_param = table_param_list[table_id];
            key_t key = rand() % table_param.max_vocabulary_size;
            key_list.push_back(key);
            dp_keys[gpu_id][lookup_id].push_back(key);
          }
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

  std::shared_ptr<HugeCTR::DataDistributor> data_distributor =
      std::make_shared<HugeCTR::DataDistributor>(core_resource_manager_list, ebc_param,
                                                 table_param_list);

  std::vector<HugeCTR::DataDistributor::Result> data_distributor_outputs;
  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    data_distributor_outputs.push_back(HugeCTR::allocate_output_for_data_distributor(
        core_resource_manager_list[gpu_id], ebc_param));
  }

  std::unique_ptr<embedding::EmbeddingCollection> ebc =
      std::make_unique<embedding::EmbeddingCollection>(resource_manager, core_resource_manager_list,
                                                       ebc_param, ebc_param, table_param_list);

  std::vector<std::vector<core23::Tensor>> sparse_dp_tensors;
  std::vector<std::vector<core23::Tensor>> sparse_dp_num_keys_per_bucket;
  std::vector<core23::Tensor> ebc_top_grads;
  std::vector<core23::Tensor> ebc_outptut;
  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());
    core23::Device device(core23::DeviceType::GPU,
                          core_resource_manager_list[gpu_id]->get_device_id());
    core23::TensorParams params = core23::TensorParams().device(device);

    std::vector<core23::Tensor> sparse_dp_tensors_on_current_gpu;
    std::vector<core23::Tensor> sparse_dp_num_keys_per_bucket_on_current_gpu;
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      auto &lookup_param = ebc_param.lookup_params[lookup_id];
      int max_hotness = lookup_param.max_hotness;
      sparse_dp_tensors_on_current_gpu.emplace_back(
          params.shape({ebc_param.universal_batch_size / num_gpus, max_hotness})
              .data_type(ebc_param.key_type));
      sparse_dp_num_keys_per_bucket_on_current_gpu.emplace_back(
          params.shape({ebc_param.universal_batch_size / num_gpus})
              .data_type(ebc_param.offset_type));
    }
    sparse_dp_tensors.push_back(sparse_dp_tensors_on_current_gpu);
    sparse_dp_num_keys_per_bucket.push_back(sparse_dp_num_keys_per_bucket_on_current_gpu);

    int64_t num_ev = 0;
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      auto &lookup_param = ebc_param.lookup_params[lookup_id];
      num_ev += (lookup_param.combiner == Combiner::Concat)
                    ? lookup_param.ev_size * lookup_param.max_hotness
                    : lookup_param.ev_size;
    }
    num_ev *= (ebc_param.universal_batch_size / num_gpus);
    ebc_top_grads.emplace_back(params.shape({num_ev}).data_type(ebc_param.emb_type));
    ebc_outptut.emplace_back(params.shape({num_ev}).data_type(ebc_param.emb_type));
  }

  auto prepare_gpu_input = [&] {
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());

      core23::copy_sync(ebc_top_grads[gpu_id], top_grads[gpu_id]);

      for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
        core23::copy_sync(sparse_dp_tensors[gpu_id][lookup_id], dp_keys[gpu_id][lookup_id]);
        core23::copy_sync(sparse_dp_num_keys_per_bucket[gpu_id][lookup_id],
                          dp_num_keys_per_bucket[gpu_id][lookup_id]);
      }
    }
  };

  auto prepare_data = [&] {
    prepare_input();
    prepare_top_grads();
    prepare_gpu_input();
  };

  auto sync_gpus = [&]() {
    for (auto core : core_resource_manager_list) {
      CudaDeviceContext __(core->get_device_id());
      HCTR_LIB_THROW(cudaStreamSynchronize(core->get_local_gpu()->get_stream()));
    }
  };
  // sync for emb table init
  sync_gpus();

  std::vector<std::vector<IGroupedEmbeddingTable *>> grouped_emb_table_ptr_list =
      ebc->get_grouped_embedding_tables();

  EmbeddingReferenceCPU<key_t, offset_t, index_t, emb_t> emb_ref{
      num_gpus, ebc_param, num_table, table_param_list, grouped_emb_table_ptr_list, output_layout};

  auto check_forward_result = [&] {
    std::cout << "compare ebc gpu emb output vs. emb reference emb output.\n";
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      std::vector<emb_t> gpu_emb_output(ebc_outptut[gpu_id].num_elements());
      core23::copy_sync(gpu_emb_output, ebc_outptut[gpu_id]);
      ASSERT_EQ(gpu_emb_output.size(), emb_ref.embedding_vec_[gpu_id].size());
      if (debug_verbose) {
        std::cout << "forward ref output:\n";
        print_array(emb_ref.embedding_vec_[gpu_id].size(), emb_ref.embedding_vec_[gpu_id]);
        std::cout << "forward gpu output:\n";
        print_array(gpu_emb_output.size(), gpu_emb_output);
      }
      float threshold = 1e-5;
      if (!std::is_same<emb_t, float>::value) threshold = 1e-4;
      assert_array_eq(gpu_emb_output.size(), gpu_emb_output, emb_ref.embedding_vec_[gpu_id],
                      threshold);
    }
    std::cout << "\t>pass compare ebc gpu emb output vs. emb reference emb output.\n";
  };

  for (int iter = 0; iter < num_iteration; ++iter) {
    std::cout << "iter:" << iter << "\n";
    prepare_data();
    sync_gpus();

    // forward
    emb_ref.embedding_forward_cpu(key_list, bucket_range);
#pragma omp parallel for num_threads(num_gpus)
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      data_distributor->distribute(gpu_id, sparse_dp_tensors[gpu_id],
                                   sparse_dp_num_keys_per_bucket[gpu_id],
                                   data_distributor_outputs[gpu_id], batch_size);
      ebc->forward_per_gpu(true, gpu_id, data_distributor_outputs[gpu_id], ebc_outptut[gpu_id],
                           batch_size);
    }
    sync_gpus();
    check_forward_result();

    // backward
    emb_ref.embedding_backward_cpu(top_grads, key_list, bucket_range);
#pragma omp parallel for num_threads(num_gpus)
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      ebc->backward_per_gpu(gpu_id, data_distributor_outputs[gpu_id], ebc_top_grads[gpu_id],
                            batch_size);
    }
    sync_gpus();

    // update
    emb_ref.embedding_update_cpu();
    sync_gpus();
#pragma omp parallel for num_threads(num_gpus)
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      ebc->update_per_gpu(gpu_id);
    }
    sync_gpus();
  }
}

// TODO: add int64_t/uint64_t test case
#define EBC_E2E_TEST_BY_TYPE(lookup_params, shard_matrix, grouped_emb_params, output_layout,     \
                             sort_strategy, allreduce_strategy)                                  \
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(                                 \
      lookup_params, shard_matrix, grouped_emb_params, output_layout, sort_strategy,             \
      allreduce_strategy);                                                                       \
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, __half>(                                \
      lookup_params, shard_matrix, grouped_emb_params, output_layout, sort_strategy,             \
      allreduce_strategy);                                                                       \
  embedding_collection_e2e<int64_t, int64_t, uint64_t, float>(lookup_params, shard_matrix,       \
                                                              grouped_emb_params, output_layout, \
                                                              sort_strategy, allreduce_strategy);

#define EBC_E2E_TEST(lookup_params, shard_matrix, grouped_emb_params)                              \
  EBC_E2E_TEST_BY_TYPE(lookup_params, shard_matrix, grouped_emb_params,                            \
                       EmbeddingLayout::FeatureMajor, SortStrategy::Radix,                         \
                       AllreduceStrategy::Dense)                                                   \
  EBC_E2E_TEST_BY_TYPE(lookup_params, shard_matrix, grouped_emb_params,                            \
                       EmbeddingLayout::FeatureMajor, SortStrategy::Radix,                         \
                       AllreduceStrategy::Sparse)                                                  \
  EBC_E2E_TEST_BY_TYPE(lookup_params, shard_matrix, grouped_emb_params,                            \
                       EmbeddingLayout::FeatureMajor, SortStrategy::Segmented,                     \
                       AllreduceStrategy::Dense)                                                   \
  EBC_E2E_TEST_BY_TYPE(lookup_params, shard_matrix, grouped_emb_params,                            \
                       EmbeddingLayout::FeatureMajor, SortStrategy::Segmented,                     \
                       AllreduceStrategy::Sparse)                                                  \
  EBC_E2E_TEST_BY_TYPE(lookup_params, shard_matrix, grouped_emb_params,                            \
                       EmbeddingLayout::BatchMajor, SortStrategy::Radix, AllreduceStrategy::Dense) \
  EBC_E2E_TEST_BY_TYPE(lookup_params, shard_matrix, grouped_emb_params,                            \
                       EmbeddingLayout::BatchMajor, SortStrategy::Radix,                           \
                       AllreduceStrategy::Sparse)                                                  \
  EBC_E2E_TEST_BY_TYPE(lookup_params, shard_matrix, grouped_emb_params,                            \
                       EmbeddingLayout::BatchMajor, SortStrategy::Segmented,                       \
                       AllreduceStrategy::Dense)                                                   \
  EBC_E2E_TEST_BY_TYPE(lookup_params, shard_matrix, grouped_emb_params,                            \
                       EmbeddingLayout::BatchMajor, SortStrategy::Segmented,                       \
                       AllreduceStrategy::Sparse)

// dp
namespace dp {
const std::vector<std::vector<int>> shard_matrix = {
    {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1},
    {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1},
};

const std::vector<GroupedTableParam> grouped_emb_params = {
    {TablePlacementStrategy::DataParallel, {0, 1, 2, 3}}};

TEST(test_embedding_collection, dp_plan0) {
  EBC_E2E_TEST(lookup_params0, shard_matrix, grouped_emb_params);
}

TEST(test_embedding_collection, dp_plan1) {
  EBC_E2E_TEST(lookup_params_with_shared_table, shard_matrix, grouped_emb_params);
}
}  // namespace dp

namespace mp {
const std::vector<std::vector<int>> shard_matrix = {
    {1, 0, 1, 1}, {0, 1, 1, 1}, {1, 0, 1, 1}, {0, 1, 1, 1},
    {1, 0, 1, 1}, {0, 1, 1, 1}, {1, 0, 1, 1}, {0, 1, 1, 1},
};

const std::vector<GroupedTableParam> grouped_emb_params = {
    {TablePlacementStrategy::ModelParallel, {0, 1, 2, 3}}};

TEST(test_embedding_collection, mp_plan0) {
  EBC_E2E_TEST(lookup_params0, shard_matrix, grouped_emb_params);
}

TEST(test_embedding_collection, mp_plan1) {
  EBC_E2E_TEST(lookup_params_with_shared_table, shard_matrix, grouped_emb_params);
}
}  // namespace mp

namespace dp_and_mp {
const std::vector<std::vector<int>> shard_matrix = {
    {1, 0, 1, 1}, {0, 1, 1, 1}, {1, 0, 1, 1}, {0, 1, 1, 1},
    {1, 0, 1, 1}, {0, 1, 1, 1}, {1, 0, 1, 1}, {0, 1, 1, 1},
};

const std::vector<GroupedTableParam> grouped_emb_params = {
    {TablePlacementStrategy::DataParallel, {2}},
    {TablePlacementStrategy::ModelParallel, {0, 1, 3}}};

TEST(test_embedding_collection, dp_and_mp_plan0) {
  EBC_E2E_TEST(lookup_params0, shard_matrix, grouped_emb_params);
}

TEST(test_embedding_collection, dp_and_mp_plan1) {
  EBC_E2E_TEST(lookup_params_with_shared_table, shard_matrix, grouped_emb_params);
}
}  // namespace dp_and_mp

namespace empty_lookup {
const std::vector<std::vector<int>> shard_matrix = {
    {1, 1, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
    {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
};

const std::vector<GroupedTableParam> grouped_emb_params = {
    {TablePlacementStrategy::ModelParallel, {0, 1, 2, 3}}};

TEST(test_embedding_collection, empty_lookup) {
  EBC_E2E_TEST(lookup_params0, shard_matrix, grouped_emb_params);
}

}  // namespace empty_lookup
