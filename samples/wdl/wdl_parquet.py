import hugectr
from mpi4py import MPI
solver = hugectr.CreateSolver(max_eval_batches = 300,
                              batchsize_eval = 16384,
                              batchsize = 16384,
                              lr = 0.001,
                              vvgpu = [[0]],
                              repeat_dataset = True,
                              i64_input_key = True)
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Parquet,
                                  source = ["./criteo_data/train/_file_list.txt"],
                                  eval_source = "./criteo_data/val/_file_list.txt",
                                  slot_size_array = [278899, 355877, 203750, 18573, 14082, 7020, 18966, 4, 6382, 1246, 49, 185920, 71354, 67346, 11, 2166, 7340, 60, 4, 934, 15, 204208, 141572, 199066, 60940, 9115, 72, 34],
                                  check_type = hugectr.Check_t.Non)
optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,
                                    update_type = hugectr.Update_t.Global,
                                    beta1 = 0.9,
                                    beta2 = 0.999,
                                    epsilon = 0.0000001)
model = hugectr.Model(solver, reader, optimizer)
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 13, dense_name = "dense",
                        data_reader_sparse_param_array = 
                        [hugectr.DataReaderSparseParam("wide_data", 1, True, 2),
                        hugectr.DataReaderSparseParam("deep_data", 2, False, 26)]))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                            workspace_size_per_gpu_in_mb = 8,
                            embedding_vec_size = 1,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding2",
                            bottom_name = "wide_data",
                            optimizer = optimizer))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                            workspace_size_per_gpu_in_mb = 114,
                            embedding_vec_size = 16,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding1",
                            bottom_name = "deep_data",
                            optimizer = optimizer))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["sparse_embedding1"],
                            top_names = ["reshape1"],
                            leading_dim=416))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["sparse_embedding2"],
                            top_names = ["reshape2"],
                            leading_dim=2))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReduceSum,
                            bottom_names = ["reshape2"],
                            top_names = ["wide_redn"],
                            axis = 1))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                            bottom_names = ["reshape1", "dense"],
                            top_names = ["concat1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["concat1"],
                            top_names = ["fc1"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc1"],
                            top_names = ["relu1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu1"],
                            top_names = ["dropout1"],
                            dropout_rate=0.5))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dropout1"],
                            top_names = ["fc2"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc2"],
                            top_names = ["relu2"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu2"],
                            top_names = ["dropout2"],
                            dropout_rate=0.5))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dropout2"],
                            top_names = ["fc3"],
                            num_output=1))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Add,
                            bottom_names = ["fc3", "wide_redn"],
                            top_names = ["add1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["add1", "label"],
                            top_names = ["loss"]))
model.compile()
model.summary()
model.fit(max_iter = 2300, display = 200, eval_interval = 1000, snapshot = 1000000, snapshot_prefix = "wdl")
