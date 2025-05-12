# dcn_parquet_train.py
import hugectr
from mpi4py import MPI
solver = hugectr.CreateSolver(max_eval_batches = 1280,
                              batchsize_eval = 1024,
                              batchsize = 1024,
                              lr = 0.001,
                              vvgpu = [[0]],
                              repeat_dataset = True)
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Parquet,
                                 check_type = hugectr.Check_t.Non,
                                 source = ["./dcn_parquet/file_list.txt"],
                                 eval_source = "./dcn_parquet/file_list_test.txt",
                                 slot_size_array = [39884, 39043, 17289, 7420, 20263, 3, 7120, 1543, 39884, 39043, 17289, 7420, 
                                                   20263, 3, 7120, 1543, 63, 63, 39884, 39043, 17289, 7420, 20263, 3, 7120, 1543 ])
optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,
                                    update_type = hugectr.Update_t.Global)
model = hugectr.Model(solver, reader, optimizer)
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 13, dense_name = "dense",
                        data_reader_sparse_param_array =
                        [hugectr.DataReaderSparseParam("data1", 1, True, 26)]))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
                           workspace_size_per_gpu_in_mb = 75,
                           embedding_vec_size = 16,
                           combiner = "sum",
                           sparse_embedding_name = "sparse_embedding1",
                           bottom_name = "data1",
                           optimizer = optimizer))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                           bottom_names = ["sparse_embedding1"],
                           top_names = ["reshape1"],
                           leading_dim=416))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                           bottom_names = ["reshape1", "dense"], top_names = ["concat1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.MultiCross,
                           bottom_names = ["concat1"],
                           top_names = ["multicross1"],
                           num_layers=6))
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
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                           bottom_names = ["dropout1", "multicross1"],
                           top_names = ["concat2"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                           bottom_names = ["concat2"],
                           top_names = ["fc2"],
                           num_output=1))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                           bottom_names = ["fc2", "label"],
                           top_names = ["loss"]))
model.compile()
model.summary()
model.graph_to_json(graph_config_file = "dcn.json")
model.fit(max_iter = 5120, display = 200, eval_interval = 1000, snapshot = 5000, snapshot_prefix = "dcn")