# Image embedding models
Get image embeddings from image embedding models.

## Supported models
- [CLIP](https://github.com/openai/CLIP)
- [DINO v2](https://github.com/facebookresearch/dinov2)
- [ImageBind](https://github.com/facebookresearch/ImageBind)

## Usage
```bash
$ make env
$ conda activate 02-image-embedding-models
$ make setup

$ python main.py
...
torch.Size([512])
torch.Size([768])
torch.Size([1024])
```

