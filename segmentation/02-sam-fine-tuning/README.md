## Before you started.
- Requirements
    - Conda
    - Download training dataset.

## Download training dataset
- https://drive.google.com/file/d/18GhVEODbTi17jSeBXdeLQ7vHPdtlTYXK/view
```
make download-dataset

```


## How to run
```
make env
conda activate 01-sam-transformer-fine-tuning
make setup
```

## How to train
```
python fine_tuning.py

ls sam_breast_cancer
config.json  pytorch_model.bin
```