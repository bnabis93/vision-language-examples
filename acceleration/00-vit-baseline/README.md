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
ViT average inference time : 4.0990471839904785ms
batch size :2
ViT average inference time : 4.2798638343811035ms
batch size :4
ViT average inference time : 4.30816650390625ms
batch size :8
ViT average inference time : 4.374895095825195ms
batch size :16
ViT average inference time : 4.983465671539307ms
batch size :32
ViT average inference time : 6.3613176345825195ms
batch size :64
ViT average inference time : 8.870742321014404ms
```
