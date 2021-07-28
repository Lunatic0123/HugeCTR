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

#include <gtest/gtest.h>
#include "utest/model_oversubscriber/mos_test_utils.hpp"
#include "HugeCTR/include/model_oversubscriber/model_oversubscriber.hpp"
#include "HugeCTR/include/parser.hpp"

using namespace HugeCTR;
using namespace mos_test;

namespace {

const char* prefix = "./model_oversubscriber_test_data/tmp_";
const char* file_list_name_train = "file_list_train.txt";
const char* file_list_name_eval = "file_list_eval.txt";
const char* snapshot_src_file = "distributed_snapshot_src";
const char* snapshot_dst_file = "distributed_snapshot_dst";
const char* keyset_file_name = "keyset_file.bin";

const int batchsize = 4096;
const long long label_dim = 1;
const long long dense_dim = 0;
const int slot_num = 128;
const int max_nnz_per_slot = 1;
const int max_feature_num = max_nnz_per_slot * slot_num;
const long long vocabulary_size = 100000;
const int emb_vec_size = 64;
const int combiner = 0;
const float scaler = 1.0f;
const int num_workers = 1;
const int num_files = 1;

const Check_t check = Check_t::Sum;
const Update_t update_type = Update_t::Local;

// const int batch_num_train = 10;
const int batch_num_eval = 1;

template <typename TypeKey>
void do_upload_and_download_snapshot(
    int batch_num_train, bool use_host_ps, bool is_distributed) {
  Embedding_t embedding_type = is_distributed ? 
                               Embedding_t::DistributedSlotSparseEmbeddingHash :
                               Embedding_t::LocalizedSlotSparseEmbeddingHash;

  // create a resource manager for a single GPU
  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back({0});
  const auto resource_manager = ResourceManagerExt::create(vvgpu, 0);

  // generate train/test datasets
  if (fs::exists(file_list_name_train)) { fs::remove(file_list_name_train); }
  if (fs::exists(file_list_name_eval))  { fs::remove(file_list_name_eval); }

  // data generation
  HugeCTR::data_generation_for_test<TypeKey, check>(file_list_name_train,
      prefix, num_files, batch_num_train * batchsize, slot_num,
      vocabulary_size, label_dim, dense_dim, max_nnz_per_slot);
  HugeCTR::data_generation_for_test<TypeKey, check>(file_list_name_eval,
      prefix, num_files, batch_num_eval * batchsize, slot_num,
      vocabulary_size, label_dim, dense_dim, max_nnz_per_slot);

  // create train/eval data readers
  const DataReaderSparseParam param = {"distributed", std::vector<int>(slot_num, max_nnz_per_slot), false, slot_num};
  std::vector<DataReaderSparseParam> data_reader_params;
  data_reader_params.push_back(param);

  std::unique_ptr<DataReader<TypeKey>> data_reader_train(new DataReader<TypeKey>(
      batchsize, label_dim, dense_dim, data_reader_params, resource_manager, true, num_workers, false));
  std::unique_ptr<DataReader<TypeKey>> data_reader_eval(new DataReader<TypeKey>(
      batchsize, label_dim, dense_dim, data_reader_params, resource_manager, true, num_workers, false));

  data_reader_train->create_drwg_norm(file_list_name_train, check);
  data_reader_eval->create_drwg_norm(file_list_name_eval, check);

  // create an embedding
  OptHyperParams hyper_params;
  hyper_params.adam.beta1 = 0.9f;
  hyper_params.adam.beta2 = 0.999f;
  hyper_params.adam.epsilon = 1e-7f;
  hyper_params.momentum.factor = 0.9f;
  hyper_params.nesterov.mu = 0.9f;

  const OptParams opt_params = {Optimizer_t::Adam,
      0.001f, hyper_params, update_type, scaler};

  const SparseEmbeddingHashParams embedding_param = {
      batchsize,       batchsize, vocabulary_size, {},        emb_vec_size,
      max_feature_num, slot_num,  combiner,        opt_params};

  auto copy = [](const std::vector<SparseTensorBag> &tensorbags, SparseTensors<TypeKey> &sparse_tensors) {
    sparse_tensors.resize(tensorbags.size());
    for(size_t j = 0; j < tensorbags.size(); ++j){
      sparse_tensors[j] = SparseTensor<TypeKey>::stretch_from(tensorbags[j]);
    }
  };
  SparseTensors<TypeKey> train_inputs;
  copy(data_reader_train->get_sparse_tensors("distributed"), train_inputs);
  SparseTensors<TypeKey> eval_inputs;
  copy(data_reader_eval->get_sparse_tensors("distributed"), eval_inputs);

  std::shared_ptr<IEmbedding> embedding = init_embedding<TypeKey, float>(
          train_inputs, eval_inputs,
          embedding_param, resource_manager, embedding_type);
  embedding->init_params();

  // train the embedding
  data_reader_train->read_a_batch_to_device();
  embedding->forward(true);
  embedding->backward();
  embedding->update_params();

  // store the snapshot from the embedding
  embedding->dump_parameters(snapshot_src_file);
  copy_sparse_model(snapshot_src_file, snapshot_dst_file);

  auto get_ext_file = [](const std::string& sparse_model_file, std::string ext) {
    return std::string(sparse_model_file) + "/" + ext;
  };

  // Make a synthetic keyset files
  std::vector<long long> keys_in_file;
  {
    size_t key_file_size_in_byte =
        fs::file_size(get_ext_file(snapshot_dst_file, "key"));
    size_t num_keys = key_file_size_in_byte / sizeof(long long);
    keys_in_file.resize(num_keys);
    std::ifstream key_ifs(get_ext_file(snapshot_dst_file, "key"));
    key_ifs.read(reinterpret_cast<char *>(keys_in_file.data()),
                                          key_file_size_in_byte);
    TypeKey *key_ptr = nullptr;
    std::vector<TypeKey> key_vec;
    if (std::is_same<TypeKey, long long>::value) {
      key_ptr = reinterpret_cast<TypeKey*>(keys_in_file.data());
    } else {
      key_vec.resize(num_keys);
      std::transform(keys_in_file.begin(), keys_in_file.end(), key_vec.begin(),
                     [](long long key) { return static_cast<unsigned>(key); });
      key_ptr = key_vec.data();
    }
    std::ofstream key_ofs(keyset_file_name, std::ofstream::binary |
                                            std::ofstream::trunc);
    key_ofs.write(reinterpret_cast<char *>(key_ptr), num_keys * sizeof(TypeKey));
  }

  std::vector<std::string> keyset_file_list;
  keyset_file_list.emplace_back(keyset_file_name);
  std::vector<std::string> sparse_embedding_files;
  sparse_embedding_files.emplace_back(snapshot_dst_file);

  // Create a ModelOversubscriber
  std::vector<std::shared_ptr<IEmbedding>> embeddings;
  embeddings.push_back(embedding);
  bool is_i64_key = std::is_same<TypeKey, unsigned>::value ? false : true;
  bool use_mixed_precision = false;

  std::shared_ptr<ModelOversubscriber> model_oversubscriber(
      new ModelOversubscriber(use_host_ps, embeddings, sparse_embedding_files,
                              resource_manager, use_mixed_precision, is_i64_key));

  Timer timer_ps;
  timer_ps.start();

  // upload embedding table from disk according to keyset
  model_oversubscriber->update(keyset_file_list);
  model_oversubscriber->dump();
  model_oversubscriber->update_sparse_model_file();

  MESSAGE_("Batch_num=" + std::to_string(batch_num_train) +
           ", embedding_vec_size=" + std::to_string(emb_vec_size) +
           ", elapsed time=" + std::to_string(timer_ps.elapsedSeconds()) + "s");

  // Check if the result is correct
  ASSERT_TRUE(check_vector_equality(snapshot_src_file, snapshot_dst_file, "key"));
  ASSERT_TRUE(check_vector_equality(snapshot_src_file, snapshot_dst_file, "emb_vector"));
  if (!is_distributed) {
    ASSERT_TRUE(check_vector_equality(snapshot_src_file, snapshot_dst_file, "slot_id"));
  }

  if (!use_host_ps) return;

  auto inc_model{model_oversubscriber->get_incremental_model(keys_in_file)};

  ASSERT_TRUE(inc_model.size() == 1);
  auto& key_vec_pair = inc_model[0];

  ASSERT_EQ(keys_in_file.size(), inc_model[0].first.size());
  ASSERT_EQ(key_vec_pair.first.size(), key_vec_pair.second.size() / emb_vec_size);

  std::string vec_file_name("./emb_vector");
  std::ofstream vec_ofs(vec_file_name, std::ofstream::binary | std::ofstream::trunc);
  vec_ofs.write(reinterpret_cast<char *>(key_vec_pair.second.data()),
      key_vec_pair.second.size() * sizeof(float));

  ASSERT_EQ(key_vec_pair.first.size(), keys_in_file.size());
  ASSERT_TRUE(check_vector_equality(snapshot_src_file, "./", "emb_vector"));
}

TEST(model_oversubscriber_test, long_long_ssd_distributed) {
  do_upload_and_download_snapshot<long long>(30, false, true);
}

TEST(model_oversubscriber_test, unsigned_host_distributed) {
  do_upload_and_download_snapshot<unsigned>(20, true, true);
}

TEST(model_oversubscriber_test, long_long_ssd_localized) {
  do_upload_and_download_snapshot<long long>(20, false, false);
}

TEST(model_oversubscriber_test, unsigned_host_localized) {
  do_upload_and_download_snapshot<unsigned>(20, true, false);
}

}  // namespace
