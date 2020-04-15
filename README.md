# Horovod Synthetic Benchmark Recreation
This is my Final Project for EECE5640 High Performance Computing taught at Northeastern 
Unviersity in the Spring 2020.

The goal of this project is to recreate the [study](https://arxiv.org/abs/1802.05799) 
done by Uber Technologies on their distributed Deep Learning training and inference framework, 
[Horovod](https://github.com/horovod/horovod). 

More specifically, we want to see that given it has been nearly 3 years since Horovod was created and
deep learning frameworks have significantly ramped up support for distributed training ever since, 
how easy/hard it is to take a Single Device training script in a supported deep learning framework
and make it distributed using Horovod and compare the process/performance of the same training script
turned distributed using the native tools provided by that framework and nothing else.

# What This Study Focuses On
The original [study](https://arxiv.org/abs/1802.05799) done by Uber Technologied focused on
comparing distributed tensorflow's performance with Horovod using 2 CNN models, Inception V3 and
ResNet-101.

This study focuses on recreating those numbers, this time with the standard VGG-16 model
provided by the framework of choice without Batch Normalization,
given how poorly it originally scaled with Horovod. Furthermore, we measure both forward and 
backwards performance instead of focusing only on forward pass performance, measured in synthetic images per
second.

It is worth mentioning that we use the newest versions of the frameworks available at the time, which are 
Tensorflow 2.1.0 and Pytorch 1.2.0. Both of these frameworks since the original study have undergone 
heavy tweaks and modifications. This is especially true for Tensorflow>1.15, in which the TapeGrad and 
newer and more diverse distributed training schemes were introduced as well as support for newer 
NVIDIA frameworks such as TensorRT and NCCL.

# What This Study Doesn't Focus On
- This study __does not__ focus on frameworks made on or leveraging other frameworks such as TensorPack,
Tensor Flow Mesh, or Keras.

- This study __does not__ focus on benchmarking optimization techniques such as adaptive learning rate.

- This study __does not__ focus on inference performance despite it being an intriguing area of investigation.

- This study __does not__ focus on other classes of DNNs such as RNNs or GANs, despite it possibly 
introducing interesting complications worth investigating.

# Studied Frameworks
The frameworks of interest in this project are [Tensorflow](https://tensorflow.org) and 
[Pytorch](https://pytorch.org).

For instructions on how to setup each framework on a cluster/slurm environment, look into the folder
specific to that framework.

# Horovod Setup on a Compute Cluster/Slurm Environment
## Pre-requisites
### NCCL

### Horovod Install Using Pip
