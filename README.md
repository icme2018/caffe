# Caffe-DICH

This repository contains code for the paper Deep Index-Compatible Hashing for Fast Image Retrieval (ICME 2018).

## Prerequisites

Linux or OSX  

NVIDIA GPU + CUDA-7.5 or CUDA-8.0 and corresponding CuDNN  

Caffe

## Modification on Caffe

* Add `src/caffe/layers/dich1_loss_layer.cpp` implementing our loss described in our paper.
* Modified `caffe.proto` 

The loss functions are implemented in `src/caffe/layers/dich1_loss_layer.cpp`. The detailed parameter settings can be found in `examples/dich_cifar/`.

