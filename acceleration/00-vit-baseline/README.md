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
# FP32
python benchmark.py
ViT average inference time : 4.16447639465332ms

# FP16
python benchmark_fp16.py
```

## Results
### FP32
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

### FP16
```
batch size :1
ViT average inference time : 4.396250247955322ms
batch size :2
ViT average inference time : 4.443643093109131ms
batch size :4
ViT average inference time : 4.431216716766357ms
batch size :8
ViT average inference time : 5.017030239105225ms
batch size :16
ViT average inference time : 5.442163944244385ms
batch size :32
ViT average inference time : 9.307048320770264ms
batch size :64
ViT average inference time : 17.756431102752686ms
```