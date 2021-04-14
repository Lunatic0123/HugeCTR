import hugectr
import sys
from mpi4py import MPI
def session_impl_test(json_file):
  solver = hugectr.CreateSolver(max_eval_batches = 2048,
                                batchsize_eval = 16384,
                                batchsize = 16384,
                                vvgpu = [[0,1,2,3,4,5,6,7]],
                                lr = 0.001,
                                i64_input_key = False,
                                use_mixed_precision = True,
                                scaler = 1024,
                                repeat_dataset = True,
                                use_cuda_graph = True)
  reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,
                                    source = ["./file_list.txt"],
                                    eval_source = "./file_list_test.txt",
                                    check_type = hugectr.Check_t.Sum)
  optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,
                                      beta1 = 0.9,
                                      beta2 = 0.999,
                                      epsilon = 0.0001)
  model = hugectr.Model(solver, reader, optimizer)
  model.construct_from_json(graph_config_file = json_file, include_dense_network = True)
  model.compile()
  model.summary()
  model.fit(max_iter = 10000, display = 200, eval_interval = 1000, snapshot = 10000, snapshot_prefix = "wdl")
  return

if __name__ == "__main__":
  json_file = sys.argv[1]
  session_impl_test(json_file)
