# xFormers ViT model Benchmark

## xFormers
- Flexible Transformers, defined by interoperable and optimized building blocks.
> xFormers is a PyTorch based library which hosts flexible Transformers parts. They are interoperable and optimized building blocks, which can be optionally be combined to create some state of the art models.

## Key features
1. Many attention mechanisms, interchangeables
    - e.g. xFormers supportted various attention mechanisms. https://facebookresearch.github.io/xformers/components/attentions.html
2. Optimized building blocks, beyond PyTorch primitives
    - e.g. fused the layer / Memory-efficient exact attention - up to 10x faster, ...
3. Benchmarking and testing tools
    - e.g. xFormers supportted benchmark / test tools. https://github.com/facebookresearch/xformers/blob/main/BENCHMARKS.md
4. Programatic and sweep friendly layer and model construction
5. Hackable
    - Not using monolithic CUDA kernels, composable building blocks
    - Using Triton for some optimized parts, explicit, pythonic and user-accessible
    - Supportted various Activation Functions. 

## Benchmark
### Setup
```
make env
conda activate 03-vit-xformers
make setup
```

### Benchmark
```
python benchmark.py

Benchmarking ViT
<torch.utils.benchmark.utils.common.Measurement object at 0x7fc42ff68a00>
profile
  Median: 5.46 ms
  IQR:    0.07 ms (5.42 to 5.49)
  365 measurements, 1 runs per measurement, 1 thread
Memory used: 786.21630859375 MB
Benchmarking Sparse ViT
<torch.utils.benchmark.utils.common.Measurement object at 0x7fc42ff1fb20>
profile
  Median: 13.90 ms
  IQR:    0.84 ms (13.54 to 14.38)
  15 measurements, 10 runs per measurement, 1 thread
Memory used: 791.16748046875 MB
```

## [Issue] Performance degradation in A100 GPU
- https://github.com/facebookresearch/xformers/blob/main/docs/source/vision_transformers.ipynb
- Vanilla Attention: 3.87ms
- Sparse Attention: 9.33ms
- Memory Efficient Attention: 6.34ms
- Sparse Attention is 2.4x slower than Vanilla Attention
- Memory Efficient Attention is 1.6x slower than Vanilla Attention
```
python reproduce.py

ViT Forward only
<torch.utils.benchmark.utils.common.Measurement object at 0x7fe7d59c9780>
profile
  Median: 3.87 ms
  IQR:    0.26 ms (3.76 to 4.02)
  514 measurements, 1 runs per measurement, 1 thread
Memory used: 28.87548828125 MB
Sparse ViT Forward only
<torch.utils.benchmark.utils.common.Measurement object at 0x7fe803ea64a0>
profile
  Median: 9.33 ms
  IQR:    0.21 ms (9.31 to 9.51)
  22 measurements, 10 runs per measurement, 1 thread
Memory used: 32.49267578125 MB
Mem efficient ViT Forward only
<torch.utils.benchmark.utils.common.Measurement object at 0x7fe803de7c10>
profile
  Median: 6.34 ms
  IQR:    0.08 ms (6.31 to 6.39)
  315 measurements, 1 runs per measurement, 1 thread
Memory used: 266.49072265625 MB
ViT average inference time : 3.186824321746826ms
ViT average inference time : 8.32324504852295ms
ViT average inference time : 5.165774822235107ms
```

## Reference
- https://github.com/facebookresearch/xformers
- https://facebookresearch.github.io/xformers/what_is_xformers.html
- https://github.com/facebookresearch/xformers/blob/main/docs/source/vision_transformers.ipynb