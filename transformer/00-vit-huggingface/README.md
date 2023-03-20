# ViT Huggingface example code
This code is an implementation of the following paper.  
[AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929.pdf)

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
$ conda activate 00-vit-huggingface

```
### Inference
```
$ python inference.py --image_path samples/test01.jpeg
Downloading (…)lve/main/config.json: 100%|███████████████████████████████████████████████████| 69.7k/69.7k [00:00<00:00, 182kB/s]
Downloading pytorch_model.bin: 100%|██████████████████████████████████████████████████████████| 346M/346M [00:07<00:00, 46.0MB/s]
Downloading (…)rocessor_config.json: 100%|██████████████████████████████████████████████████████| 160/160 [00:00<00:00, 33.4kB/s]
logit shape : torch.Size([1, 1000])
Predictions class : Siberian husky
```

## Docker
### Build
```
$ make build
$ docker images
REPOSITORY                                           TAG                                       IMAGE ID       CREATED          SIZE
00-vit-huggingface                                   latest                                    2b6795a26c0d   42 seconds ago   17.5GB
```

### Run the docker images
```
$ make run-docker
$ python3 inference.py --image_path samples/test01.jpeg
Downloading (…)lve/main/config.json: 100%|███████████████████████████████████████████████████| 69.7k/69.7k [00:00<00:00, 182kB/s]
Downloading pytorch_model.bin: 100%|██████████████████████████████████████████████████████████| 346M/346M [00:07<00:00, 46.0MB/s]
Downloading (…)rocessor_config.json: 100%|██████████████████████████████████████████████████████| 160/160 [00:00<00:00, 33.4kB/s]
logit shape : torch.Size([1, 1000])
Predictions class : Siberian husky
```