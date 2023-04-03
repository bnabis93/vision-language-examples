# ViT Huggingface example code
This code is an implementation of the following paper.  
[AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929.pdf)
- ViT pytorch cifar10 training code

## Server spec
- GPU : NVIDIA GeForce RTX 2080 Ti
- CPU : AMD Ryzen Threadripper 2950X 16-Core Processo

## ViT Paper Review (Korean)
- https://bono93.notion.site/ViT-AN-IMAGE-IS-WORTH-16X16-WORDS-TRANSFORMERS-FOR-IMAGE-RECOGNITION-AT-SCALE-42d67a0f8925402ba3b1124cd558a11b

## Prerequisities
- Conda or Docker

## Conda
### Setting
```
$ make env
$ conda activate 01-vit-pytorch-train
$ make setup
```

### Training
```
$ python train.py
Files already downloaded and verified
Files already downloaded and verified
Epoch 1/10
----------
Train Epoch: 0
Train Epoch: 0 [0/50000 (0%)]   Loss: 2.533456
Train Epoch: 0 [3200/50000 (6%)]        Loss: 1.910644
Train Epoch: 0 [6400/50000 (13%)]       Loss: 2.203146
...
```