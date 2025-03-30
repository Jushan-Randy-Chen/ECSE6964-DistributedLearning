# Final project for ECSE 6964 Distributed Optimization and Learning at Rensselaer Polytechnic Institute
This repository is adapted from the official code implementation for the paper [ICML 2022] Generalization of Decentralized SGD. 
In this project, we extend the original paper to the time-varying communication topology.

## [ICML 2022] Generalization of Decentralized SGD

The repository contains the offical implementation of the paper

> [**ICML 2022**] [**Topology-aware Generalization of Decentralized SGD**](https://arxiv.org/pdf/2206.12680.pdf)

This paper studies the algorithmic stability and generalizability of decentralized stochastic gradient descent (D-SGD). We prove that the consensus model learned by D-SGD is $\mathcal{O}{(N^{-1}+m^{-1} +\lambda^2)}$-stable in expectation in the non-convex non-smooth setting, where $N$ is the total sample size of the whole system, $m$ is the worker number, and $1-\lambda$ is the spectral gap that measures the connectivity of the communication topology. These results then deliver an $\mathcal{O}{(N^{-(1+\alpha)/2}+m^{-(1+\alpha)/2}+\lambda^{1+\alpha} + \phi_{\mathcal{S}})}$ in-average generalization bound, which is non-vacuous even when $\lambda$ is closed to $1$, in contrast to vacuous as suggested by existing literature on the projected version of D-SGD. Our theory indicates that the generalizability of D-SGD has a positive correlation with the spectral gap, and can explain why consensus control in initial training phase can ensure better generalization. Experiments of VGG-11 and ResNet-18 on CIFAR-10, CIFAR-100 and Tiny-ImageNet justify our theory. To our best knowledge, this is **the first work on the topology-aware generalization analysis of vanilla D-SGD**.

Please kindly refer to our [**arXiv version**](https://arxiv.org/pdf/2206.12680.pdf) and [**presentation slides**](https://github.com/Raiden-Zhu/Generalization-of-DSGD/blob/main/Slides_ICML2022_Generalization_of_D_SGD.pdf) for the latest updates and more detailed information.

![image](https://github.com/Raiden-Zhu/Generalization-of-DSGD/blob/main/Poster_ICML2022_Generalization_of_D_SGD_v8.png)

## Environment

```
numpy>=1.21.2

Pillow>=9.2.0

torch>=1.10.1

torchvision>=0.11.2
```

## Dataset
The Tiny Imagenet dataset can be downloaded from [here](https://paperswithcode.com/dataset/tiny-imagenet). The CIFAR-10 and CIFAR-100 datasets can be downloaded automatically by ```torchvision.datasets``` when running the code.
In addition, one can use ```--path```(default is "../data") to set the location of the dataset folder.

## Example of usage
Train ResNet-18 on CIFAR-10 dataset with fully-connected topology:
```
python gpu_work.py --seed 555 --mode "all" --size 64 --batch_size 64 --learning_rate 0.4   --model_name "ResNet18" --dataset_name "CIFAR10" --milestones 2400 4800 --early_stop 6000 --num_epoch 6000 --gpu True
```
Train ResNet-18 on CIFAR-100 dataset with ring topology:
```
python gpu_work.py --seed 555 --mode "ring" --size 64 --batch_size 64 --learning_rate 0.4   --model_name "ResNet18" --dataset_name "CIFAR100" --milestones 4000 8000 --early_stop 10000 --num_epoch 10000 --gpu True
```
Train ResNet-18 on Tiny ImageNet dataset with grid topology:
```
python gpu_work.py --seed 555 --mode "meshgrid" --size 64 --batch_size 64 --learning_rate 0.4   --model_name "ResNet18" --dataset_name "TinyImageNet" --milestones 4000 8000 --early_stop 10000 --num_epoch 10000 --gpu True
```
Train VGG-11 on CIFAR-10 dataset with exponential topology:
```
python gpu_work.py --seed 555 --mode "exponential" --size 64 --batch_size 64 --learning_rate 0.4   --model_name "VGG11BN" --dataset_name "TinyImageNet" --milestones 8000 16000 --early_stop 20000 --num_epoch 20000 --gpu True


