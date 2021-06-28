"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sparse_operation_kit.kit_lib import get_nccl_unique_id, gen_random_seed, plugin_init

from tensorflow.python.ops import collective_ops
try:
    from tensorflow.distribute import MultiWorkerMirroredStrategy
except:
    from tensorflow.distribute.experimental import MultiWorkerMirroredStrategy
from tensorflow.distribute import MirroredStrategy, get_replica_context, has_strategy, get_strategy
from tensorflow import constant, TensorShape, function
from tensorflow.dtypes import int32, int64
from tensorflow import print as tf_print

def Init(**kwargs):
    """
    This function is used to do the initialization for plugin.
    It should only be called once for this process.
    And it must be called under the tf.distribute.Strategy.Scope().
    """
    @function
    def _single_worker_init(**kwargs):
        replica_ctx = get_replica_context()
        replica_ctx.merge_call(lambda strategy: 
            tf_print("You are using the plugin with MirroredStrategy."))
        nccl_unique_id = replica_ctx.merge_call(lambda strategy:
                    get_nccl_unique_id())
        global_random_seed = replica_ctx.merge_call(lambda strategy:
                    gen_random_seed())

        global_id = replica_ctx.replica_id_in_sync_group
        status = plugin_init(global_id, replica_ctx.num_replicas_in_sync, nccl_unique_id, global_random_seed,
                             global_batch_size=kwargs['global_batch_size']) #TODO: input from kwargs
        return status

    @function
    def _multi_worker_init(**kwargs):
        replica_ctx = get_replica_context()
        global_id = replica_ctx.replica_id_in_sync_group
        task_id = replica_ctx.strategy.cluster_resolver.task_id
        if task_id == 0 and global_id == 0:
            unique_id = get_nccl_unique_id()
            re = collective_ops.broadcast_send(unique_id,
                                                TensorShape([32,]),
                                                int32,
                                                group_size=replica_ctx.num_replicas_in_sync,
                                                group_key=1,
                                                instance_key=2,
                                                timeout=10)
        else:
            re = collective_ops.broadcast_recv(TensorShape([32,]),
                                                int32,
                                                group_size=replica_ctx.num_replicas_in_sync,
                                                group_key=1,
                                                instance_key=2,
                                                timeout=10)
        if task_id == 0 and global_id == 0:
            global_seed = gen_random_seed()
            re_seed = collective_ops.broadcast_send(global_seed,
                                                TensorShape([1,]),
                                                int64,
                                                group_size=replica_ctx.num_replicas_in_sync,
                                                group_key=1,
                                                instance_key=3,
                                                timeout=10)
        else:
            re_seed = collective_ops.broadcast_recv(TensorShape([1,]),
                                                int64,
                                                group_size=replica_ctx.num_replicas_in_sync,
                                                group_key=1,
                                                instance_key=3,
                                                timeout=10)

        status = plugin_init(global_id, replica_ctx.num_replicas_in_sync, re, re_seed, 
                             global_batch_size=kwargs['global_batch_size']) #TODO: input from kwargs
        return status

    if has_strategy():
        strategy = get_strategy()
        if isinstance(strategy, MirroredStrategy):
            return strategy.run(_single_worker_init, kwargs=kwargs)
        elif isinstance(strategy, MultiWorkerMirroredStrategy):
            return strategy.run(_multi_worker_init, kwargs=kwargs)
        else:
            raise RuntimeError("This strategy type is not supported yet.")
    else:
        raise RuntimeError("This function must be called inside tf.distribute.Strategy.Scope().")
    
