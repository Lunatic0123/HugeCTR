"""
 Copyright (c) 2023, NVIDIA CORPORATION.
 
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

import tensorrt as trt
import ctypes

import os
import numpy as np
import struct

import json
import pytest

import onnx_graphsurgeon as gs
from onnx import shape_inference
import onnx

import pycuda.driver as cuda
import pycuda.autoinit
import onnxruntime as ort

import torch
from torch.utils.data import DataLoader

plugin_lib_name = "/usr/local/hps_trt/lib/libhps_plugin.so"
handle = ctypes.CDLL(plugin_lib_name, mode=ctypes.RTLD_GLOBAL)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_ENGINE_ONNX_SAVED_PATH = "dlrm_pytorch_with_hps.trt"
BZ = 1024
KEY_DTYPE = np.int32
TARGET_DTYPE = np.float32

args = dict()

args["gpu_num"] = 1  # the number of available GPUs
args["iter_num"] = 50  # the number of training iteration
args["slot_num"] = 26  # the number of feature fields in this embedding layer
args["embed_vec_size"] = 128  # the dimension of embedding vectors
args["dense_dim"] = 13  # the dimension of dense features
args["global_batch_size"] = 1024  # the globally batchsize for all GPUs
args["max_vocabulary_size"] = 260000
args["vocabulary_range_per_slot"] = [[i * 10000, (i + 1) * 10000] for i in range(26)]
args["combiner"] = "mean"

args["ps_config_file"] = "dlrm_pytorch.json"
args["embedding_table_path"] = "dlrm_pytorch_sparse.model"
args["onnx_path"] = "dlrm_pytorch.onnx"
args["modified_onnx_path"] = "dlrm_pytorch_with_hps.onnx"
args["np_key_type"] = np.int32
args["np_vector_type"] = np.float32

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args["gpu_num"])))

hps_config = {
    "supportlonglong": False,
    "models": [
        {
            "model": "dlrm",
            "sparse_files": ["dlrm_pytorch_sparse.model"],
            "num_of_worker_buffer_in_pool": 3,
            "embedding_table_names": ["sparse_embedding0"],
            "embedding_vecsize_per_table": [128],
            "maxnum_catfeature_query_per_table_per_sample": [26],
            "default_value_for_each_table": [1.0],
            "deployed_device_list": [0],
            "max_batch_size": 1024,
            "cache_refresh_percentage_per_iteration": 0.2,
            "hit_rate_threshold": 1.0,
            "gpucacheper": 1.0,
            "gpucache": True,
        }
    ],
}
hps_config_json_object = json.dumps(hps_config, indent=4)
with open(args["ps_config_file"], "w") as outfile:
    outfile.write(hps_config_json_object)


def generate_random_samples(
    num_samples, vocabulary_range_per_slot, dense_dim, key_dtype=args["np_key_type"]
):
    keys = list()
    for vocab_range in vocabulary_range_per_slot:
        keys_per_slot = np.random.randint(
            low=vocab_range[0], high=vocab_range[1], size=(num_samples, 1), dtype=key_dtype
        )
        keys.append(keys_per_slot)
    keys = np.concatenate(np.array(keys), axis=1)
    numerical_features = np.random.random((num_samples, dense_dim)).astype(np.float32)
    labels = np.random.randint(low=0, high=2, size=(num_samples, 1))
    return keys, numerical_features, labels


import torch


class MLP(torch.nn.Module):
    def __init__(self, arch, name, out_activation, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.mlp = torch.nn.Sequential()

        for idx in range(1, len(arch) - 1):
            self.mlp.add_module(
                name + "_linear_layer_%d" % idx, torch.nn.Linear(arch[idx - 1], arch[idx])
            )
            self.mlp.add_module(name + "_relu_layer_%d" % idx, torch.nn.ReLU(inplace=True))

        idx = len(arch) - 1
        if out_activation == "relu":
            self.mlp.add_module(
                name + "_linear_layer_%d" % idx, torch.nn.Linear(arch[idx - 1], arch[idx])
            )
            self.mlp.add_module(name + "_relu_layer_%d" % idx, torch.nn.ReLU(inplace=True))
        elif out_activation == "sigmoid":
            self.mlp.add_module(
                name + "_linear_layer_%d" % idx, torch.nn.Linear(arch[idx - 1], arch[idx])
            )
            self.mlp.add_module(name + "_relu_layer_%d" % idx, torch.nn.Sigmoid())

    def forward(self, x):
        y = self.mlp(x)
        return y


class SecondOrderFeatureInteraction(torch.nn.Module):
    def __init__(self):
        super(SecondOrderFeatureInteraction, self).__init__()

    def forward(self, inputs, num_feas):
        dot_products = torch.reshape(
            torch.matmul(inputs, torch.transpose(inputs, 1, 2)), (-1, num_feas * num_feas)
        )
        indices = torch.tensor([i * num_feas + j for j in range(1, num_feas) for i in range(j)])
        flat_interactions = torch.index_select(dot_products, 1, indices)
        return flat_interactions


class DLRM(torch.nn.Module):
    def __init__(
        self, init_tensors, embed_vec_size, slot_num, dense_dim, arch_bot, arch_top, **kwargs
    ):
        super(DLRM, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(init_tensors, freeze=False)

        self.embed_vec_size = embed_vec_size
        self.slot_num = slot_num
        self.dense_dim = dense_dim
        self.arch_bot = arch_bot
        self.arch_top = arch_top

        self.bot_mlp = MLP([self.dense_dim] + arch_bot, name="bottom", out_activation="relu")
        self.interaction_layer = SecondOrderFeatureInteraction()
        self.interaction_out_dim = self.slot_num * (self.slot_num + 1) // 2
        self.top_mlp = MLP(
            [self.interaction_out_dim + self.arch_bot[-1]] + arch_top,
            name="top",
            out_activation="sigmoid",
        )

    def forward(self, inputs):
        categorical_features = inputs[0]
        numerical_features = inputs[1]

        embedding_vector = self.embedding(categorical_features)
        dense_x = self.bot_mlp(numerical_features)

        concat_features = torch.concat(
            [embedding_vector, torch.reshape(dense_x, (-1, 1, self.arch_bot[-1]))], 1
        )

        Z = self.interaction_layer(concat_features, self.slot_num + 1)
        z = torch.concat([dense_x, Z], 1)
        logit = self.top_mlp(z)
        return logit


def train(args):
    init_tensors = torch.Tensor(
        np.ones(
            shape=[args["max_vocabulary_size"], args["embed_vec_size"]],
            dtype=args["np_vector_type"],
        )
    )

    model = DLRM(
        init_tensors,
        args["embed_vec_size"],
        args["slot_num"],
        args["dense_dim"],
        arch_bot=[512, 256, args["embed_vec_size"]],
        arch_top=[1024, 1024, 512, 256, 1],
    )

    print(model)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    keys, numerical_features, labels = generate_random_samples(
        args["global_batch_size"] * args["iter_num"],
        args["vocabulary_range_per_slot"],
        args["dense_dim"],
        args["np_key_type"],
    )
    x0_iterator = iter(
        DataLoader(
            torch.from_numpy(keys),
            batch_size=args["global_batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
    )
    x1_iterator = iter(
        DataLoader(
            torch.from_numpy(numerical_features),
            batch_size=args["global_batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
    )
    y_iterator = iter(
        DataLoader(
            torch.from_numpy(labels).float(),
            batch_size=args["global_batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
    )

    for i in range(args["iter_num"]):
        inputs = [next(x0_iterator), next(x1_iterator)]
        labels = next(y_iterator)
        preds = model(inputs)
        loss = criterion(preds.squeeze(), labels.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("-" * 20, "Step {}, loss: {}".format(i, loss), "-" * 20)
    return model


def convert_to_sparse_model(embeddings_weights, embedding_table_path, embedding_vec_size):
    os.system("mkdir -p {}".format(embedding_table_path))
    with open("{}/key".format(embedding_table_path), "wb") as key_file, open(
        "{}/emb_vector".format(embedding_table_path), "wb"
    ) as vec_file:
        for key in range(embeddings_weights.shape[0]):
            vec = embeddings_weights[key]
            key_struct = struct.pack("q", key)
            vec_struct = struct.pack(str(embedding_vec_size) + "f", *vec)
            key_file.write(key_struct)
            vec_file.write(vec_struct)


def torch_to_onnx(trained_model):
    dummy_keys = torch.randint(
        0,
        args["max_vocabulary_size"],
        (args["global_batch_size"], args["slot_num"]),
        dtype=torch.int32,
    )
    dummy_numerical_features = torch.randn(args["global_batch_size"], args["dense_dim"])
    torch.onnx.export(
        trained_model,
        [dummy_keys, dummy_numerical_features],
        args["onnx_path"],
        verbose=True,
        input_names=["keys", "numerical_features"],
        output_names=["output"],
        dynamic_axes={"keys": {0: "batch_size"}, "numerical_features": {0: "batch_size"}},
    )


def onnx_graph_surgery():
    graph = gs.import_onnx(onnx.load("dlrm_pytorch.onnx"))
    saved = []

    for node in graph.nodes:
        if node.name == "/embedding/Gather":
            categorical_features = gs.Variable(
                name="categorical_features", dtype=np.int32, shape=("unknown", 26)
            )
            hps_node = gs.Node(
                op="HPS_TRT",
                attrs={
                    "ps_config_file": "dlrm_pytorch.json\0",
                    "model_name": "dlrm\0",
                    "table_id": 0,
                    "emb_vec_size": 128,
                },
                inputs=[categorical_features],
                outputs=[node.outputs[0]],
            )
            graph.nodes.append(hps_node)
            saved.append(categorical_features)
            node.outputs.clear()
    for i in graph.inputs:
        if i.name == "numerical_features":
            saved.append(i)
    graph.inputs = saved

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), "dlrm_pytorch_with_hps.onnx")


def build_engine_from_onnx(onnx_model_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        EXPLICIT_BATCH
    ) as network, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser, builder.create_builder_config() as builder_config:
        model = open(onnx_model_path, "rb")
        parser.parse(model.read())

        profile = builder.create_optimization_profile()
        profile.set_shape("categorical_features", (1, 26), (1024, 26), (1024, 26))
        profile.set_shape("numerical_features", (1, 13), (1024, 13), (1024, 13))
        builder_config.add_optimization_profile(profile)
        engine = builder.build_serialized_network(network, builder_config)
        return engine


def create_hps_plugin_creator():
    trt_version = [int(n) for n in trt.__version__.split(".")]
    handle = ctypes.CDLL(plugin_lib_name, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    plg_registry = trt.get_plugin_registry()
    hps_plugin_creator = plg_registry.get_plugin_creator("HPS_TRT", "1", "")
    return hps_plugin_creator


def create_hps_plugin(hps_plugin_creator):
    ps_config_file = trt.PluginField(
        "ps_config_file",
        np.array(["dlrm_pytorch.json\0"], dtype=np.string_),
        trt.PluginFieldType.CHAR,
    )
    model_name = trt.PluginField(
        "model_name", np.array(["dlrm\0"], dtype=np.string_), trt.PluginFieldType.CHAR
    )
    table_id = trt.PluginField("table_id", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32)
    emb_vec_size = trt.PluginField(
        "emb_vec_size", np.array([128], dtype=np.int32), trt.PluginFieldType.INT32
    )
    params = trt.PluginFieldCollection([ps_config_file, model_name, table_id, emb_vec_size])
    hps_plugin = hps_plugin_creator.create_plugin(name="hps", field_collection=params)
    return hps_plugin


def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    trt_engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return trt_engine


def predict(
    categorical_h_input,
    numerical_h_input,
    h_output,
    categorical_d_input,
    numerical_d_input,
    d_output,
    stream,
    bindings,
    context,
):
    cuda.memcpy_htod_async(categorical_d_input, categorical_h_input, stream)
    cuda.memcpy_htod_async(numerical_d_input, numerical_h_input, stream)
    context.execute_async_v2(bindings, stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return h_output


def create_onnx_inference():
    sess = ort.InferenceSession("dlrm_pytorch.onnx")
    return sess


def run_onnx_inference(sess, categorical_h, numerical_h):
    res = sess.run(
        output_names=[sess.get_outputs()[0].name],
        input_feed={
            sess.get_inputs()[0].name: categorical_h,
            sess.get_inputs()[1].name: numerical_h,
        },
    )
    return res


def test_build_engine_for_pytorch():
    trained_model = train(args)
    embedding_weights = trained_model.state_dict()["embedding.weight"]
    convert_to_sparse_model(embedding_weights, args["embedding_table_path"], args["embed_vec_size"])

    torch_to_onnx(trained_model)
    onnx_graph_surgery()

    # hps_plugin_creator = create_hps_plugin_creator()
    # hps_plugin = create_hps_plugin(hps_plugin_creator)

    serialized_engine = build_engine_from_onnx("dlrm_pytorch_with_hps.onnx")
    with open("dlrm_pytorch_with_hps.trt", "wb") as fout:
        fout.write(serialized_engine)
    print("Successfully build the TensorRT engine")
    assert True


def test_run_engine():
    # hps_plugin_creator = create_hps_plugin_creator()
    # hps_plugin = create_hps_plugin(hps_plugin_creator)

    sess = create_onnx_inference()

    engine = load_engine(TRT_ENGINE_ONNX_SAVED_PATH)
    context = engine.create_execution_context()
    context.set_input_shape("categorical_features", (BZ, 26))
    context.set_input_shape("numerical_features", (BZ, 13))

    categorical_h_input = np.random.randint(0, 1, (BZ, 26)).astype(KEY_DTYPE)
    numerical_h_input = np.random.random((BZ, 13)).astype(TARGET_DTYPE)
    h_output = np.empty([BZ, 1], dtype=TARGET_DTYPE)

    categorical_d_input = cuda.mem_alloc(1 * categorical_h_input.nbytes)
    numerical_d_input = cuda.mem_alloc(1 * numerical_h_input.nbytes)
    d_output = cuda.mem_alloc(1 * h_output.nbytes)

    bindings = [int(categorical_d_input), int(numerical_d_input), int(d_output)]
    stream = cuda.Stream()

    for i in range(5):
        categorical_h_input = np.random.randint(0, 1, (BZ, 26)).astype(KEY_DTYPE)
        numerical_h_input = np.random.random((BZ, 13)).astype(TARGET_DTYPE)
        preds = predict(
            categorical_h_input,
            numerical_h_input,
            h_output,
            categorical_d_input,
            numerical_d_input,
            d_output,
            stream,
            bindings,
            context,
        )

        onnx_preds = run_onnx_inference(sess, categorical_h_input, numerical_h_input)
        diff = preds.flatten() - onnx_preds[0].flatten()
        mse = np.mean(diff * diff)
        print(mse)
        print(preds)
        print(onnx_preds[0])
        assert mse <= 1e-6
