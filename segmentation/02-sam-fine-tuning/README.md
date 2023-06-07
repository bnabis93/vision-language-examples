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

## How to train
```
python fine_tuning.py

ls sam_breast_cancer
config.json  pytorch_model.bin
```