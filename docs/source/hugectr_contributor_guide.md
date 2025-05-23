# Contributing to HugeCTR

```{contents}
---
depth: 2
local: true
backlinks: none
---
```

## Overview of Contributing to HugeCTR

We're grateful for your interest in HugeCTR and value your contributions. You can contribute to HugeCTR by:
* submitting a [feature, documentation, or bug request](https://github.com/NVIDIA/HugeCTR/issues/new/choose).

  **NOTE**: After we review your request, we'll assign it to a future release. If you think the issue should be prioritized over others, comment on the issue.

* proposing and implementing a new feature.

  **NOTE**: Once we agree to the proposed design, you can go ahead and implement the new feature using the steps outlined in the [Contribute New Code section](#contribute-new-code).

* implementing a pending feature or fixing a bug.

  **NOTE**: Use the steps outlined in the [Contribute New Code section](#contribute-new-code). If you need more information about a particular issue,
  add your comments on the issue.

## Contribute New Code

1. Build HugeCTR or Sparse Operation Kit (SOK) from source using the steps outlined in the [Set Up the Development Environment with Merlin Containers](#set-up-the-development-environment-with-merlin-containers).
2. [File an issue](https://github.com/NVIDIA/HugeCTR/issues/new/choose) and add a comment stating that you'll work on it.
3. Start coding.

   **NOTE**: Don't forget to add or update the unit tests properly.

4. [Create a pull request](https://github.com/nvidia/HugeCTR/compare) for you work.
5. Wait for a maintainer to review your code.

   You may be asked to make additional edits to your code if necessary. Once approved, a maintainer will merge your pull request.

If you have any questions or need clarification, don't hesitate to add comments to your issue and we'll respond promptly.

## How to Start your Development

### Set Up the Development Environment With Merlin Containers

We provide options to disable the installation of HugeCTR and HugeCTR Triton Backend in [Merlin Dockerfiles](https://github.com/NVIDIA-Merlin/Merlin/tree/main/docker) so that our contributors can build the development environment (container) from them. By simply clone the code of HugeCTR into this environment and build you can start the journey of development.

**Note**: the message on terminal below is not errors if you are working in such containers.
```
groups: cannot find name for group ID 1007
I have no name!@56a762eae3f8:/hugectr
```

In [Merlin CTR Dockerfile](https://github.com/NVIDIA-Merlin/Merlin/blob/main/docker/dockerfile.ctr), [Merlin Tensorflow Dockerfile](https://github.com/NVIDIA-Merlin/Merlin/blob/main/docker/dockerfile.tf), we provide a set of arguments to setup your HugeCTR development container:

The arguments and configurations in this example can be used in all the three containers building:

```
docker build --pull -t ${DST_IMAGE} -f ${DOCKER_FILE} --build-arg RELEASE=false --build-arg RMM_VER=vnightly --build-arg CUDF_VER=vnightly --build-arg NVTAB_VER=vnightly --build-arg HUGECTR_DEV_MODE=true --no-cache .
```

For RMM_VER, CUDF_VER, NVTAB_VER, you can simply specify the release tag e.g. `v1.0` or `vnightly` if you want to build with the head of the `main` branch.  With specifying HUGECTR_DEV_MODE=true, you can disable HugeCTR installation.

**Docker CLI Quick Reference**
```
$ docker build [<opts>] <path> | <URL>
               Build a new image from the source code at PATH
  -f, --file path/to/Dockerfile
               Path to the Dockerfile to use. Default: Dockerfile.
  --build-arg <varname>=<value>
               Name and value of a build argument defined with ARG
               Dockerfile instruction
  -t "<name>[:<tag>]"
               Repository names (and optionally with tags) to be applied
               to the resulting image
  --label =<label>
               Set metadata for an image
  -q, --quiet  Suppress the output generated by containers
  --rm         Remove intermediate containers after a successful build
```

### Build HugeCTR Training Container from Source

To build HugeCTR Training Container from source, do the following:

1. Build the `hugectr:devel` image using the steps outlined [here](#set-up-the-development-environment-with-merlin-containers). Remember that this instruction is only for the [Merlin CTR Dockerfile](https://github.com/NVIDIA-Merlin/Merlin/blob/main/docker/dockerfile.ctr).


2. Download the HugeCTR repository and the third-party modules that it relies on by running the following commands:
   ```shell
   $ git clone https://github.com/NVIDIA/HugeCTR.git
   $ cd HugeCTR
   $ git submodule update --init --recursive
   ```

3. Build HugeCTR from scratch using one or any combination of the following options:
   - **SM**: You can use this option to build HugeCTR with a specific compute capability (DSM=90) or multiple compute capabilities (DSM="80;70"). The default compute capability
     is 70, which uses the NVIDIA V100 GPU. For more information, refer to the [Compute Capability](hugectr_user_guide.md#compute-capability) table.
   - **CMAKE_BUILD_TYPE**: You can use this option to build HugeCTR with Debug or Release. When using Debug to build, HugeCTR will print more verbose logs and execute GPU tasks
     in a synchronous manner.
     average of eval_batches results. Only one thread and chunk will be used in the data reader. Performance will be lower when in validation mode. This option is set to OFF by
     default.
   - **ENABLE_MULTINODES**: You can use this option to build HugeCTR with multiple nodes. This option is set to OFF by default. For more information, refer to the [deep and cross network samples](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/samples/dcn) directory on GitHub.
   - **ENABLE_HDFS**: You can use this option to build HugeCTR together with HDFS to enable HDFS related functions. Permissible values are `ON` and `OFF` *(default)*. Setting this option to `ON` leads to installing all necessary Hadoop modules that are required for building so that it can connect to HDFS deployments.
   - **ENABLE_S3**: You can use this option to build HugeCTR together with Amazon AWS S3 SDK to enable S3 related functions. Permissible values are `ON` and `OFF` *(default)*. Setting this option to `ON` leads to building all necessary AWS SKKs and dependencies that are required for building AND running both HugeCTR and S3. 

   **Please note that setting DENABLE_HDFS=ON or DENABLE_S3=ON requires root permission. So before using these two options to do the customized building, make sure you use `-u root` when you run the docker container.**

   Here are some examples of how you can build HugeCTR using these build options:
   ```shell
   $ mkdir -p build && cd build
   $ cmake -DCMAKE_BUILD_TYPE=Release -DSM=80 .. # Target is NVIDIA A100
   $ make -j && make install
   ```

   ```shell
   $ mkdir -p build && cd build
   $ cmake -DCMAKE_BUILD_TYPE=Release -DSM="80;90" -DENABLE_MULTINODES=ON .. # Target is NVIDIA A100 / H100 with the multi-node mode on.
   $ make -j && make install
   ```

   ```shell
   $ mkdir -p build && cd build
   $ cmake -DCMAKE_BUILD_TYPE=Release -DSM="70;80" -DENABLE_HDFS=ON .. # Target is NVIDIA V100 / A100 with HDFS components mode on.
   $ make -j && make install
   ```

   ```shell
   $ mkdir -p build && cd build
   $ cmake -DCMAKE_BUILD_TYPE=Debug -DSM="70;80" .. # Target is NVIDIA V100 / A100 with Debug mode.
   $ make -j && make install
   ```
   By default, HugeCTR is installed at /usr/local. However, you can use CMAKE_INSTALL_PREFIX to install HugeCTR to non-default location:
   ```shell
   $ cmake -DCMAKE_INSTALL_PREFIX=/opt/HugeCTR -DSM=70 ..
   ```

### Build Sparse Operation Kit (SOK) from Source

To build the Sparse Operation Kit component in HugeCTR, do the following:

1. Build the `hugectr:tf-plugin` docker image using the steps noted [here](#set-up-the-development-environment-with-merlin-containers). Remember that this instruction is only for the [Merlin Tensorflow Dockerfile](https://github.com/NVIDIA-Merlin/Merlin/blob/main/docker/dockerfile.tf).


2. Download the HugeCTR repository by running the following command:
   ```shell
   $ git clone https://github.com/NVIDIA/HugeCTR.git hugectr
   ```

3. Build and install libraries to the system paths by running the following commands:
   ```shell
   $ cd hugectr/sparse_operation_kit
   $ python setup.py install
   ```

   You can config different environment variables for compiling SOK, please refer to [this section](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/env_vars/env_vars.html) for more details.
