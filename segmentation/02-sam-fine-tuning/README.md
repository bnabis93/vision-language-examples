## Before you started.
- Requirements
    - Conda
    - Download training dataset.

## Download training dataset and model
- https://drive.google.com/file/d/18GhVEODbTi17jSeBXdeLQ7vHPdtlTYXK/view
```
make download-dataset
make download-model
```

## How to run
```
make env
conda activate 01-sam-transformer-fine-tuning
make setup
```

## Create embedding for training
```
python create_embedding.py
```

## Before training 
- you should change sam github code.
- `segment_anything/modeling/mask_decoder.py", line 127`
```python 
# Expand per-image data in batch direction to be per-mask
if image_embeddings.shape[0] != tokens.shape[0]: 
    src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
else: 
    src = image_embeddings 
# src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)

```

## How to train
```
python fine_tuning.py

ls sam_breast_cancer
config.json  pytorch_model.bin
```