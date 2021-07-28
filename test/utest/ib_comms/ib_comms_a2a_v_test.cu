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

#ifdef ENABLE_MPI
#include <random>
#include "HugeCTR/include/collectives/ib_comm.hpp"
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/tensor2.hpp"
#include "HugeCTR/include/general_buffer2.hpp"
#include "HugeCTR/include/general_buffer2.hpp"
#include "HugeCTR/include/utils.hpp"
#include "HugeCTR/include/resource_managers/resource_manager_ext.hpp"
#include "utest/test_utils.h"
#include "gtest/gtest.h"

using namespace HugeCTR;

namespace {

  template <bool is_integral, typename T> struct uniform_distribution_selector;
  template <typename T> struct uniform_distribution_selector<true, T>
  {
    using type = typename std::uniform_int_distribution<T>;
  };
  template <typename T> struct uniform_distribution_selector<false, T>
  {
    using type = typename std::uniform_real_distribution<T>;
  };
  template <typename T>
  using uniform_distribution_t = typename uniform_distribution_selector<std::is_integral<T>::value, T>::type;

  template <typename TypeEmbeddingComp>
  struct IbCommsTest
  {
    public:
      IbCommsTest(const std::vector<int> &device_list, size_t max_size) :
        num_gpus_(device_list.size()),
        max_size_(max_size) {

          MPI_Comm_size(MPI_COMM_WORLD, &num_procs_);

          // Align max_size
          max_elems_per_dest_ = max_size_ / (num_procs_ * num_gpus_) / sizeof(TypeEmbeddingComp);
          max_size_ = (max_elems_per_dest_) * (num_procs_ * num_gpus_) * sizeof(TypeEmbeddingComp);
          max_elems_ = max_size_ / sizeof(TypeEmbeddingComp);
          max_elems_per_gpu_ = max_elems_ / device_list.size();
          max_size_per_gpu_ = max_elems_per_gpu_ * sizeof(TypeEmbeddingComp);
          max_size_ = max_size_per_gpu_ * num_gpus_;
          max_elems_per_proc_ = max_elems_ / num_procs_;

          std::vector<std::vector<int>> vvgpu;
          for (int i = 0; i < num_procs_; i++) {
            vvgpu.push_back(device_list);
          }
          resource_manager_ = ResourceManagerExt::create(vvgpu, 0, DeviceMap::LOCAL_FIRST);
          ib_comm_ = resource_manager_->get_ib_comm();

          init_buffers();
          gen_uniform_size(max_size_);
        }

      ~IbCommsTest()
      {
        ib_comm_->finalize();
      }

      void gen_uniform_size(size_t total_send_size)
      {
        size_t num_dest = num_gpus_ * num_procs_;
        size_t send_size_per_dst = total_send_size / (num_gpus_ * num_procs_);
        // Align to element type
        send_size_per_dst = (send_size_per_dst / sizeof(TypeEmbeddingComp)) * sizeof(TypeEmbeddingComp);
        auto& device_list = resource_manager_->get_local_gpu_device_id_list();

        for (size_t g = 0; g < num_gpus_; g++) {
          size_t* h_send_size_ptr = h_send_sizes_[g].get_ptr();
          size_t* h_recv_size_ptr = h_recv_sizes_[g].get_ptr();
          for (size_t d = 0; d < num_dest; d++) {
            h_send_size_ptr[d] = send_size_per_dst;
            h_recv_size_ptr[d] = send_size_per_dst;
          }
          CK_CUDA_THROW_(cudaSetDevice(device_list[g]));
          CK_CUDA_THROW_(cudaMemcpy(d_send_sizes_[g].get_ptr(), h_send_sizes_[g].get_ptr(),
                h_send_sizes_[g].get_num_elements()*sizeof(size_t),
                cudaMemcpyHostToDevice));
          CK_CUDA_THROW_(cudaMemcpy(d_recv_sizes_[g].get_ptr(), h_recv_sizes_[g].get_ptr(),
                h_recv_sizes_[g].get_num_elements()*sizeof(size_t),
                cudaMemcpyHostToDevice));
        }
      }

      void gen_rand_size()
      {
        size_t num_dest = num_gpus_ * num_procs_;
        std::default_random_engine generator;
        uniform_distribution_t<size_t> distribution(1, max_elems_per_dest_);
        
        auto& device_list = resource_manager_->get_local_gpu_device_id_list();

        for (size_t g = 0; g < num_gpus_; g++) {
          size_t* h_send_size_ptr = h_send_sizes_[g].get_ptr();
          size_t* h_recv_size_ptr = h_recv_sizes_[g].get_ptr();
          for (size_t d = 0; d < num_dest; d++) {
            h_send_size_ptr[d] = distribution(generator)*sizeof(TypeEmbeddingComp);
          }
          CK_MPI_THROW_(MPI_Alltoall(
                h_send_size_ptr, sizeof(size_t)*num_gpus_, MPI_BYTE,
                h_recv_size_ptr, sizeof(size_t)*num_gpus_, MPI_BYTE,
                MPI_COMM_WORLD));
          
          CK_CUDA_THROW_(cudaSetDevice(device_list[g]));
          CK_CUDA_THROW_(cudaMemcpy(d_send_sizes_[g].get_ptr(), h_send_sizes_[g].get_ptr(),
                h_send_sizes_[g].get_num_elements()*sizeof(size_t),
                cudaMemcpyHostToDevice));
          CK_CUDA_THROW_(cudaMemcpy(d_recv_sizes_[g].get_ptr(), h_recv_sizes_[g].get_ptr(),
                h_recv_sizes_[g].get_num_elements()*sizeof(size_t),
                cudaMemcpyHostToDevice));
        }
      }

      void fill_buffers()
      {
        std::default_random_engine generator;
        uniform_distribution_t<TypeEmbeddingComp> distribution(1,100);
        // reset recv buffers
        for (size_t g = 0; g < num_gpus_; g++) {
          memset(h_recv_buffs_[g].get_ptr(), 0, max_elems_*sizeof(TypeEmbeddingComp));
          memset(h_recv_buffs_out_[g].get_ptr(), 1, max_elems_*sizeof(TypeEmbeddingComp));
        }
        
        for (size_t g = 0; g < num_gpus_; g++) {
          for (size_t s = 0; s < max_elems_; s++) {
            TypeEmbeddingComp number = distribution(generator);
            *(h_send_buffs_[g].get_ptr() + s) = number;
          }
        }

        // for (size_t g = 0; g < num_gpus_; g++) {
        //   for (size_t s = 0; s < max_elems_; s++) {
        //     *(h_send_buffs_[g].get_ptr() + s) = s;
        //   }
        // }

        auto& device_list = resource_manager_->get_local_gpu_device_id_list();
        for (size_t g = 0; g < num_gpus_; g++) {
          CK_CUDA_THROW_(cudaSetDevice(device_list[g]));
          CK_CUDA_THROW_(cudaMemcpy(d_send_buffs_[g].get_ptr(), h_send_buffs_[g].get_ptr(),
                max_elems_*sizeof(TypeEmbeddingComp), cudaMemcpyHostToDevice));
          CK_CUDA_THROW_(cudaMemcpy(d_recv_buffs_[g].get_ptr(), h_recv_buffs_[g].get_ptr(),
                max_elems_*sizeof(TypeEmbeddingComp), cudaMemcpyHostToDevice));
        }

        for (size_t g = 0; g < num_gpus_; g++) {
          CK_CUDA_THROW_(cudaSetDevice(device_list[g]));
          ib_comm_->update_a2a_coll_sizes(coll_handle_, 
              d_send_sizes_[g].get_ptr(),
              d_recv_sizes_[g].get_ptr(),
              0, g);
        }
      }

      void do_device_a2a()
      {
        auto& device_list = resource_manager_->get_local_gpu_device_id_list();
        for (size_t g = 0; g < num_gpus_; g++) {
          CK_CUDA_THROW_(cudaSetDevice(device_list[g]));
          ib_comm_->post_send_command_a2a<TypeEmbeddingComp>(coll_handle_, 0, g);
        }
        for (size_t g = 0; g < num_gpus_; g++) {
          CK_CUDA_THROW_(cudaSetDevice(device_list[g]));
          CK_CUDA_THROW_(cudaDeviceSynchronize());
        }
      }

      void do_host_a2a()
      {
        std::vector<MPI_Request> send_requests(num_procs_*num_gpus_);
        std::vector<MPI_Request> recv_requests(num_procs_*num_gpus_);
        std::vector<MPI_Status>   statuses(num_procs_*num_gpus_);
        std::vector<MPI_Datatype> send_dtypes(num_procs_*num_gpus_);
        std::vector<MPI_Datatype> recv_dtypes(num_procs_*num_gpus_);

        for (size_t g = 0; g < num_gpus_; g++) {
          for (int r = 0; r < num_procs_; r++) {

            size_t offset = g*num_procs_ + r;

            std::vector<MPI_Aint> displacements;
            displacements.resize(num_gpus_);
            for (size_t d = 0; d < num_gpus_; d++) {
              displacements[d] = MPI_Aint(d*max_elems_per_dest_*sizeof(TypeEmbeddingComp));
            }

            std::vector<int> h_send_sizes_int_;
            std::vector<int> h_recv_sizes_int_;
            for (size_t s = 0; s < num_gpus_; s++) {
              auto send_sizes = h_send_sizes_[g].get_ptr() + (r*num_gpus_);
              auto recv_sizes = h_recv_sizes_[g].get_ptr() + (r*num_gpus_);

              h_send_sizes_int_.push_back(int(send_sizes[s]));
              h_recv_sizes_int_.push_back(int(recv_sizes[s]));
            }

            std::vector<MPI_Datatype> in_types(num_gpus_, MPI_BYTE);
            MPI_Type_create_struct(num_gpus_, h_send_sizes_int_.data(),
                displacements.data(), in_types.data(), &send_dtypes[offset]);
            MPI_Type_create_struct(num_gpus_, h_recv_sizes_int_.data(),
                displacements.data(), in_types.data(), &recv_dtypes[offset]) ;
            MPI_Type_commit( &send_dtypes[offset] );
            MPI_Type_commit( &recv_dtypes[offset] );

            CK_MPI_THROW_(MPI_Isend(h_send_buffs_[g].get_ptr() + (r*num_gpus_*max_elems_per_dest_),
                  1, send_dtypes[offset], r, g, MPI_COMM_WORLD, &send_requests[offset]));
            CK_MPI_THROW_(MPI_Irecv(h_recv_buffs_[g].get_ptr() + (r*num_gpus_*max_elems_per_dest_),
                  1, recv_dtypes[offset], r, g, MPI_COMM_WORLD, &recv_requests[offset]));
          }
        }
        CK_MPI_THROW_(MPI_Waitall(num_procs_*num_gpus_, send_requests.data(), statuses.data()));
        CK_MPI_THROW_(MPI_Waitall(num_procs_*num_gpus_, recv_requests.data(), statuses.data()));
      }

      void compare_host_and_device()
      {
        for (size_t g = 0; g < num_gpus_; g++) {
          CK_CUDA_THROW_(cudaMemcpy(h_recv_buffs_out_[g].get_ptr(), d_recv_buffs_[g].get_ptr(),
                max_elems_*sizeof(TypeEmbeddingComp), cudaMemcpyDeviceToHost));
        }

        for (size_t g = 0; g < num_gpus_; g++) {
          for (size_t e = 0; e < max_elems_; e++) {
            if (*(h_recv_buffs_[g].get_ptr() + e) != *(h_recv_buffs_out_[g].get_ptr() + e)) {
              size_t my_proc = resource_manager_->get_process_id();
              std::cout << my_proc << ": Data mismatch at gpu " << g << " element: " << e << 
                " expected: " << *(h_recv_buffs_[g].get_ptr() + e) << " got: " << *(h_recv_buffs_out_[g].get_ptr() + e) << std::endl;
              exit(1);
            }
          }
        }
      }

    private:

      size_t num_gpus_;
      size_t max_size_;
      size_t max_elems_;
      size_t max_elems_per_gpu_;
      size_t max_elems_per_proc_;
      size_t max_elems_per_dest_;
      size_t max_size_per_gpu_;
      int    num_procs_ = 1;

      std::shared_ptr<ResourceManager> resource_manager_;
      IbComm* ib_comm_; // TODO: Make it shared so we have only one instance of ibcomm
      HierA2AvCollHandle coll_handle_;

      std::vector<Tensor2<TypeEmbeddingComp>> h_send_buffs_;
      std::vector<Tensor2<TypeEmbeddingComp>> h_recv_buffs_;

      std::vector<Tensor2<TypeEmbeddingComp>> h_recv_buffs_out_;

      std::vector<Tensor2<TypeEmbeddingComp>> d_send_buffs_;
      std::vector<Tensor2<TypeEmbeddingComp>> d_recv_buffs_;

      std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> dev_bufs_;
      std::vector<std::shared_ptr<GeneralBuffer2<CudaHostAllocator>>> host_bufs_;

      std::vector<Tensor2<size_t>>  h_send_sizes_;
      std::vector<Tensor2<size_t>>  h_recv_sizes_;

      std::vector<Tensor2<size_t>>  d_send_sizes_;
      std::vector<Tensor2<size_t>>  d_recv_sizes_;

      void init_buffers() 
      {
        coll_handle_ = ib_comm_->register_hier_a2a_v_coll();
        h_send_buffs_.resize(num_gpus_);
        h_recv_buffs_.resize(num_gpus_);
        h_recv_buffs_out_.resize(num_gpus_);
        d_send_buffs_.resize(num_gpus_);
        d_recv_buffs_.resize(num_gpus_);

        dev_bufs_.resize(num_gpus_);
        host_bufs_.resize(num_gpus_);

        h_send_sizes_.resize(num_gpus_);
        d_send_sizes_.resize(num_gpus_);
        h_recv_sizes_.resize(num_gpus_);
        d_recv_sizes_.resize(num_gpus_);

        CudaDeviceContext context;
        for (size_t g = 0; g < num_gpus_; g++) {
          auto& device_list = resource_manager_->get_local_gpu_device_id_list();
          context.set_device(device_list[g]);
          dev_bufs_[g] = GeneralBuffer2<CudaAllocator>::create();
          host_bufs_[g] = GeneralBuffer2<CudaHostAllocator>::create();

          dev_bufs_[g]->reserve({max_elems_}, &d_send_buffs_[g]);
          dev_bufs_[g]->reserve({max_elems_}, &d_recv_buffs_[g]);
          dev_bufs_[g]->reserve({num_gpus_*num_procs_}, &d_send_sizes_[g]);
          dev_bufs_[g]->reserve({num_gpus_*num_procs_}, &d_recv_sizes_[g]);
          dev_bufs_[g]->allocate();

          host_bufs_[g]->reserve({max_elems_}, &h_send_buffs_[g]);
          host_bufs_[g]->reserve({max_elems_}, &h_recv_buffs_[g]);
          host_bufs_[g]->reserve({max_elems_}, &h_recv_buffs_out_[g]);
          host_bufs_[g]->reserve({num_gpus_*num_procs_}, &h_send_sizes_[g]);
          host_bufs_[g]->reserve({num_gpus_*num_procs_}, &h_recv_sizes_[g]);
          host_bufs_[g]->allocate();

          ib_comm_->set_a2a_coll_buf(coll_handle_, 
              (void*)d_send_buffs_[g].get_ptr(), max_elems_ * sizeof(TypeEmbeddingComp),
              (void*)d_recv_buffs_[g].get_ptr(), max_elems_ * sizeof(TypeEmbeddingComp),
              g);
        }
        ib_comm_->register_a2a_coll_buf(coll_handle_);
        ib_comm_->set_ready_to_transfer();
      }
  };

  template <typename TypeEmbeddingComp>
  void test_ib_comm(const std::vector<int> &device_list) 
  {
    int num_procs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (num_procs == 1) return;

    const size_t MAX_SIZE = 16*1024*1024;
    IbCommsTest<TypeEmbeddingComp> test(device_list, MAX_SIZE);

    // Uniform size test
    for (size_t size = 1024; size < MAX_SIZE; size *= 2) {
      test.gen_uniform_size(size);
      test.fill_buffers();
      test.do_host_a2a();
      test.do_device_a2a();
      test.compare_host_and_device();
    }

    // Random size test
    for (int i = 0; i < 10; i++) {
      test.gen_rand_size();
      test.fill_buffers();
      test.do_host_a2a();
      test.do_device_a2a();
      test.compare_host_and_device();
    }
  }
} // namespace

TEST(ib_comms_a2a_v_test, fp_1gpu_per_node)  { test_ib_comm<float>({0}); }
TEST(ib_comms_a2a_v_test, u16_4gpu_per_node) { test_ib_comm<uint16_t>({0,2,4,7}); }
TEST(ib_comms_a2a_v_test, fp_8gpu_per_node)  { test_ib_comm<float>({0,1,2,3,4,5,6,7}); }
#endif
