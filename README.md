# Caffe-DICH

This repository contains code for the paper Deep Index-Compatible Hashing for Fast Image Retrieval (ICME 2018).

## Prerequisites

Linux or OSX

NVIDIA GPU + CUDA-7.5 or CUDA-8.0 and corresponding CuDNN

Caffe

## Modification on Caffe

* Add `src/caffe/layers/dich1_loss_layer.cpp` implementing our loss described in our paper. The `scale` in the code is the `$\alpha$` in our loss function.
* Add `examples/dich_cifar/*.prototxt` describing the structure and hyper-parameters of the net.

## Datasets

We use NUS-WIDE and CIFAR-10 datasets in our experiments. You can get NUS-WIDE dataset [here](http://lms.comp.nus.edu.sg/research/NUS-WIDE.htm). As far as CIFAR-10, you can download it and format the data by simply running the following commands:
```
cd $CAFFE_ROOT
./data/cifar10/get_cifar10.sh
./examples/cifar10/create_cifar10.sh
```

## Compiling

The compiling process is the same with caffe. You can refer to Caffe installation instructions [here](http://caffe.berkeleyvision.org/installation.html).

## Training

You can create `train.prototxt` to describe the structure of the net and `solver.prototxt` to define the hyper-parameter of the net as we defined in `examples/dich_cifar/`. Then you can train the model for each dataset and specify the log file using the following command
```
./build/tools/caffe train --solver=solver*.prototxt -weights ./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu gpu_id 2>&1 | tee -a log.txt
```
