# HugeCTR to ONNX Converter #
hugectr2onnx is a python package that can convert HugeCTR models to ONNX format. It can improve the compatibility of HugeCTR with other deep learning frameworks given that Open Neural Network Exchange (ONNX) serves as an open-source format for AI models.

To use the ONNX converter, you need to prepare HugeCTR models files and graph configuration files which can be obtained with HugeCTR training APIs. Please refer to [HugeCTR Python Interface](https://nvidia-merlin.github.io/HugeCTR/master/api/python_interface.html#) to get familiar with how to train and save your models. To learn the workflow of training HugeCTR model, converting to ONNX and making inference with ONNX Runtime, please refer to [HugeCTR to ONNX Demo Notebook](../notebooks/hugectr2onnx_demo.ipynb).

## Overview ##
**HugeCTR Model Format**

The trained models with HugeCTR consist of the dense model and the sparse models. There will be one binary dense model file which stores the weights of all the dense layers. The number of the sparse model folders will be equal to the that of embedding table, and there exist binary files for keys and embedding vectors in each folder. All these model files will be saved automatically according to the `snapshot` and `snapshot_prefix` values within the HugeCTR training API `hugectr.Model.fit()`. A graph configuration JSON file is required to load these binary model files correctly, which can be derived with the HugeCTR training API `hugectr.Model.graph_to_json()`. Here is an example of HugeCTR trained models and graph configuration JSON file:

```bash
wdl_model
├── wdl0_sparse_2000.model
│   ├── emb_vector
│   └── key
├── wdl1_sparse_2000.model
│   ├── emb_vector
│   └── key
├── wdl_dense_2000.model
└── wdl.json
```

**ONNX Converter Features**

The HugeCTR to ONNX converter will parse the graph configuration JSON file for each HugeCTR layer and construct an equivalent ONNX graph with ONNX operators. At the same time, the converter will read the HugeCTR model files and upload the weights to the corresponding ONNX node if the HugeCTR layer has some weights. We also provide an option for users to choose whether to convert the sparse models to ONNX or not, given that the sparse embedding model is usually very large and the weight alignment is relatively easily compared to the dense part.

If you choose to convert both dense and sparse models to ONNX, the converted ONNX graph will take dense features and sparse keys as the inputs and expose prediction result as the output. If you only convert the dense model to ONNX, then the converted ONNX graph will require dense features and sparse embedding vectors as the inputs. The following figure depicts the differences of the converted ONNX models in these two cases (the red dotted frames represent lookup and reduction for sparse embedding vectors).

<div align=center><img src ="readme_src/wdl_onnx.png" width="800"/></div>
<div align=center>Fig. 1: The ONNX graph of WDL model. Left: converting both dense and sparse models. Right: converting only the dense model. </div>

## Installation ##
There are several ways to install this package.

**Use NGC Container**

In the docker image: `nvcr.io/nvidia/merlin/merlin-training:22.04`, hugectr2onnx is already installed and you can directly import this package via:
```python
import hugectr2onnx
```
    
**Build from source**

If you want to build this package from the source code, here are the steps to follow:
```shell
$ git clone https://github.com/NVIDIA/HugeCTR.git
$ cd HugeCTR/onnx_converter
$ python3 setup.py install
```

## API Docs ##
**hugectr2onnx.converter.convert(\*args, \*\*kwargs)**

This function is used to convert the HugeCTR models to ONNX format. It requires graph configuration JSON file and model files from HugeCTR training, and will save the converted ONNX model to the specified path.

***Parameters***

* onnx_model_path (string): the path to store the ONNX model.

* graph_config (string): the graph configuration JSON file of the HugeCTR model.

* dense_model (string): the file of the dense weights for the HugeCTR model.

* convert_embedding (boolean): whether to convert the sparse embeddings for the HugeCTR model (optional).

* sparse_models (List[str]): the files of the sparse embeddings for the HugeCTR model (optional).

* ntp_file (string): the file of the non-trainable parameters for the HugeCTR model (optional).

* graph_name (string): the graph name for the ONNX model (optional)

***Examples***

```python
hugectr2onnx.converter.convert(onnx_model_path = "wdl.onnx",
                            graph_config = "wdl.json",
                            dense_model = "wdl_dense_2000.model",
                            convert_embedding = True,
                            sparse_models = ["wdl0_sparse_2000.model", "wdl1_sparse_2000.model"])
```

**Note**: When making inference using the converted ONNX model, the categorical keys from the Parquet dataframe should be offsetted with the same `slot_size_array` as HugeCTR training before being fed into the ONNX inference session. For more details, please refer to [Parquet Dataset](https://nvidia-merlin.github.io/HugeCTR/master/api/python_interface.html#dataset-formats).

## Layer Support ##

Most HugeCTR layers are supported in the HugeCTR to ONNX converter, and there are a few unsupported layers because they have not been extended to the general usage of HugeCTR model training.

* Supported Layers
  * [Add Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#add-layer)
  * [BatchNorm Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#batchnorm-layer)
  * [Concat Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#concat-layer)
  * [Dropout Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#dropout-layer)
  * [ElementwiseMultiply Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#elementwisemultiply-layer)
  * [ELU Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#elu-layer)
  * [FmOrder2 Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#fmorder2-layer)
  * [FullyConnected Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#fullyconnected-layer)
  * [FusedFullyConnected Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#fusedfullyconnected-layer)
  * [FusedReshapeConcat Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#fusedreshapeconcat-layer)
  * [Interaction Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#interaction-layer)
  * [MatrixMultiply Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#matrixmutiply-layer)
  * [MultiCross Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#multicross-layer)
  * [PReLU_Dice Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#preludice-layer)
  * [ReduceMean Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#reducemean-layer)
  * [ReduceSum Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#reducesum-layer)
  * [ReLU Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#relu-layer)
  * [Reshape Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#reshape-layer)
  * [Scale Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#scale-layer)
  * [Sigmoid Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#sigmoid-layer)
  * [Slice Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#slice-layer)
  * [Softmax Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#softmax-layer)
  * [Sub Layer](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#sub-layer)
  * [WeightMultiply](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#weightmultiply-layer)
  * [BinaryCrossEntropyLoss](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#binarycrossentropyloss)
  * [CrossEntropyLoss](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#crossentropyloss)
  * [MultiCrossEntropyLoss](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#multicrossentropyloss)
* Unsupported Layers
  * Cast
  * [FusedReshapeConcatGeneral](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#fusedreshapeconcatgeneral-layer)
  * [GRU](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#gru-layer)
  * [Gather](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/source/api/hugectr_layer_book.md#gather-layer)
  * ReLUHalf

**Note**: The BinaryCrossEntropyLoss layer will be replaced by a Sigmoid layer when converting the HugeCTR model graph to ONNX. The CrossEntropyLoss layer will be replaced by a Softmax layer when converting the HugeCTR model graph to ONNX. The MultiCrossEntropyLoss layer will be replaced by a Sigmoid layer when converting the HugeCTR model graph to ONNX.

