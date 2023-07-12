# ViT Baseline

## Environment setup
- Python 3.10, Conde env
```
make env
conda activate 00-vit-baseline
make setup
```

## Benchmark
- GPU: Nvidia A100
```
python benchmark.py
ViT average inference time : 4.16447639465332ms
```

## Results
```
batch size :1
ViT average inference time : 12.45668649673462ms
batch size :2
ViT average inference time : 17.118918895721436ms
batch size :4
ViT average inference time : 26.85886859893799ms
batch size :8
ViT average inference time : 47.40768909454346ms
batch size :16
ViT average inference time : 93.63363981246948ms
batch size :32
ViT average inference time : 171.22406482696533ms
```
