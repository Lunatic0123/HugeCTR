#
# Copyright (c) 2021, NVIDIA CORPORATION.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sparse_operation_kit.core.embedding_variable import EmbeddingVariable
from sparse_operation_kit.kit_lib import create_embedding_dense, plugin_dense_fprop
from sparse_operation_kit.embeddings import embedding_ops

import tensorflow as tf

class All2AllDenseEmbedding(tf.keras.layers.Layer):
    """
    Abbreviated as ``sok.All2AllDenseEmbedding(*args, **kwargs)``.

    This is a wrapper class for all2all dense embedding layer.
    It can be used to create a dense embedding layer which will distribute
    keys based on `gpu_id = key % gpu_num` to each GPU.

    Parameters
    ----------
    max_vocabulary_size_per_gpu: integer
            the first dimension of embedding variable whose shape is 
            [max_vocabulary_size_per_gpu, embedding_vec_size].
    embedding_vec_size: integer
            the second dimension of embedding variable whose shape is 
            [max_vocabulary_size_per_gpu, embedding_vec_size].
    slot_num: integer
            the number of feature-fileds which will be processed at the same time in
            each iteration, where all feature-fileds produce embedding vectors
            of the same dimension.
    nnz_per_slot: integer
            the number of valid keys in each slot. The number of valid keys in each slot 
            is the same.

    Examples
    --------
    .. code-block:: python

        emb_layer = sok.All2AllDenseEmbedding(max_vocabulary_size_per_gpu, 
                                              embedding_vec_size, 
                                              slot_num, nnz_per_slot)
        
        @tf.function
        def _train_step(inputs, labels):
            emb_vectors = emb_layer(inputs)
            ...
        
        for i, (inputs, labels) in enumerate(dataset):
            _train_step(inputs)
    """
    def __init__(self,
                 max_vocabulary_size_per_gpu,
                 embedding_vec_size, 
                 slot_num,
                 nnz_per_slot,
                 **kwargs):
        super(All2AllDenseEmbedding, self).__init__(**kwargs)

        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self.embedding_vec_size = embedding_vec_size
        self.slot_num = slot_num
        self.nnz_per_slot = nnz_per_slot

        self.var = EmbeddingVariable.CreateInstances(shape=[self.max_vocabulary_size_per_gpu, self.embedding_vec_size],
                                                     trainable=True)

        self.emb = create_embedding_dense(self.var.values[0].emb_handle,
                                          input_dispatcher="All2AllInput",
                                          embedding_lookuper="dense_gather",
                                          output_dispatcher="All2AllOutput",
                                          slot_num=self.slot_num,
                                          nnz_per_slot=self.nnz_per_slot)

    @property
    def embedding_variable(self):
        return self.var

    @tf.function
    def call(self, inputs, training=True):
        """
        The forward logic of this wrapper class.

        Parameters
        ----------
        inputs: tf.Tensor
                keys are stored in tf.Tensor. It must be stored in row-major.
                The shape of tf.Tensor does not matter.
        training: boolean
                whether training or not.

        Returns
        -------
        emb_vector: tf.float
                the embedding vectors for the input keys. Its shape is
                *[batchsize, slot_num, nnz_per_slot, embedding_vec_size]*
        """
        emb_vector = plugin_dense_fprop(self.emb,
                                        self.var,
                                        values=inputs,
                                        global_replica_id=embedding_ops.get_global_replica_id(),
                                        training=training, 
                                        unique_op_name=self.var.name)
        return emb_vector

