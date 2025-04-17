# System requirements

## Operating System
nnssl so far has only been tested on Linux (Ubuntu 22.04). However due to it being closely tied to Windows and MacOS it should (hopefully) work there out of the box!
In case it does not, feel free to raise a github issue and we will try to help you out.

## Hardware requirements
We strongly recommend using GPU, due to pre-training often times being computationally expensive. 
Theoretically CPU and Apple M1/M2 (currently Apple mps does not implement 3D convolutions, so you might have to use the CPU on those devices) as devices are also supported but it does not really make sense to use them for pre-training purposes, so do with that what you will.

### Hardware requirements for Training
We recommend you to use a data-center grade GPU akin to a A100-40GB for training. This will allow you to train the majority of the models implemented in nnssl.
While some of the Trainers implemented may also work for smaller GPUs with 24GB or 32GB we currently don't have an overview table to give a comprehensive overview, so expect to run into OOM errors for some of these. To keep your GPU occupied during pre-training we also recommend a strong CPU to go along with the GPU. 12 cores (24 threads) should be on the lower end, but this will always depend on the amount of CPU load in the augmentation of the pre-training scheme which can vary a lot!

### Hardware Requirements for inference
As opposed to nnU-Net nnssl *does not support inference* as the pretext tasks used for pre-training are generally not intended for inference. 
Should you find pre-text tasks for which this would make sense, please open a github issue and detail the benefit this may have, then we might consider providing this.
If you want to use the pre-trained models for downstream tasks, please check out our [segmentation adaptation repo](https://github.com/TaWald/nnUNet/tree/nnssl_finetuning_inclusion).
This repository contains an adaptation of nnU-Net which allows you to use the pre-trained models from nnssl for any nnU-Net downstream task!


### Example hardware configurations (from nnU-Net)
Example workstation configurations for training:
- CPU: Ryzen 5800X - 5900X or 7900X would be even better! We have not yet tested Intel Alder/Raptor lake but they will likely work as well.
- GPU: RTX 3090 or RTX 4090
- RAM: 64GB
- Storage: SSD (M.2 PCIe Gen 3 or better!)

Example Server configuration for training:
- CPU: 2x AMD EPYC7763 for a total of 128C/256T. 16C/GPU are highly recommended for fast GPUs such as the A100!
- GPU: 8xA100 PCIe (price/performance superior to SXM variant + they use less power)
- RAM: 1 TB
- Storage: local SSD storage (PCIe Gen 3 or better) or ultra fast network storage

(nnssl by default uses one GPU per training, however it also supports multi-GPU training. We will provide more documentation on this in the future.)

### Setting the correct number of Workers for data augmentation (training only)
Identically to nnU-Net you will need to manually set the number of processes nnU-Net uses for data augmentation according to your 
CPU/GPU ratio. For the server above (256 threads for 8 GPUs), a good value would be 24-30. You can do this by 
setting the `nnUNet_n_proc_DA` environment variable (`export nnUNet_n_proc_DA=XX`). 
Recommended values (assuming a recent CPU with good IPC) are 10-12 for RTX 2080 ti, 12 for a RTX 3090, 16-18 for 
RTX 4090, 28-32 for A100. Optimal values may vary depending on the number of input channels/modalities and number of classes.

# Installation instructions
We strongly recommend that you install nnssl in a virtual environment! Pip or anaconda are both fine. If you choose to 
compile PyTorch from source (see below), you will need to use conda instead of pip. 

Use a recent version of Python! 3.12 or newer is guaranteed to work!

**nnssl v2 can coexist with nnU-Net v1! Both can be installed at the same time.**

Currenly we don't support a direct pip installation of nnssl. Instead you will need to clone the repository and install it manually. 
To do so, please follow the instructions below.

```bash
conda create -n nnssl python=3.12
conda activate nnssl

git clone https://github.com/MIC-DKFZ/nnssl
cd nnssl
pip install -e . 
```

Installing nnssl will add several new commands to your terminal. These commands are used to run the entire nnssl
pipeline. You can execute them from any location on your system. All nnssl commands have the prefix `nnssl_` for
easy identification.

Note that these commands simply execute python scripts. If you installed nnssl in a virtual environment, this
environment must be activated when executing the commands. You can see what scripts/functions are executed by 
checking the `project.scripts` in the [pyproject.toml](../pyproject.toml) file.

All nnssl commands have a `-h` option which gives information on how to use them.
