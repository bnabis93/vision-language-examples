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
ViT average inference time : 4.308726787567139ms
batch size :2
ViT average inference time : 4.269990921020508ms
batch size :4
ViT average inference time : 4.266459941864014ms
batch size :8
ViT average inference time : 4.643599987030029ms
batch size :16
ViT average inference time : 5.297105312347412ms
batch size :32
ViT average inference time : 9.313437938690186ms
batch size :64
ViT average inference time : 17.998650074005127ms
batch size :128
ViT average inference time : 35.26674509048462ms
batch size :256
ViT average inference time : 69.79572057723999ms
batch size :512
ViT average inference time : 140.56599617004395ms
```