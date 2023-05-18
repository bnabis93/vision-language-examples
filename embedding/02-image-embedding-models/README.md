# Image embedding models
Get image embeddings from image embedding models.

## Supported models
- [CLIP](https://github.com/openai/CLIP)
- [DINO v2](https://github.com/facebookresearch/dinov2)
- [ImageBind](https://github.com/facebookresearch/ImageBind)

## How to use it?
### Setup
```bash
$ make env
$ conda activate 02-image-embedding-models
$ make setup
```
### Download dataset
- We use Animal-10 dataset for embedding visulization
- Download link: https://www.kaggle.com/datasets/alessiocorrado99/animals10

### Run the code
```
# Verify all models are working
$ python embedding_models.py
...
torch.Size([512])
torch.Size([768])
torch.Size([1024])
```
- Visualization
    - CLIP take a time 5 minutes to run
    - Dino v2 take a time 30 minutes to run
```
$ python clip_vis.py
$ ls
clip_tsne.png

$ python dinov2_vis.py
$ ls
dinov2_tsne.png

$ python imagebind_vis.py
$ ls
imagebind_tsne.png
```

