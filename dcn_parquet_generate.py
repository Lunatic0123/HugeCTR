# dcn_parquet_generate.py
import hugectr
from hugectr.tools import DataGeneratorParams, DataGenerator
data_generator_params = DataGeneratorParams(
  format = hugectr.DataReaderType_t.Parquet,
  label_dim = 1,
  dense_dim = 13,
  num_slot = 26,
  i64_input_key = False,
  source = "./dcn_parquet/file_list.txt",
  eval_source = "./dcn_parquet/file_list_test.txt",
  slot_size_array = [39884, 39043, 17289, 7420, 20263, 3, 7120, 1543, 39884, 39043, 17289, 7420, 
                     20263, 3, 7120, 1543, 63, 63, 39884, 39043, 17289, 7420, 20263, 3, 7120,
                     1543 ],
  dist_type = hugectr.Distribution_t.PowerLaw,
  power_law_type = hugectr.PowerLaw_t.Short)
data_generator = DataGenerator(data_generator_params)
data_generator.generate()