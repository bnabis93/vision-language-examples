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
ViT average inference time : 5.244340896606445ms
batch size :2
ViT average inference time : 6.640884876251221ms
batch size :4
ViT average inference time : 10.941822528839111ms
batch size :8
ViT average inference time : 19.58521842956543ms
batch size :16
ViT average inference time : 37.01387882232666ms
batch size :32
ViT average inference time : 68.04256677627563ms
batch size :64
ViT average inference time : 131.94079399108887ms
```
