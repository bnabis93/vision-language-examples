# Acceleration
Methods for transformer acceleration.

## Benchmark results
- xFormers has a inference speed degradation in A100 GPU.
- https://github.com/bnabis93/vision-language-examples/issues/14
- In FP16, a faster transformer is a good candidate.
- In FP16, trt conversion may be slower above a certain batch size.

### FP32
|                                 | batch size=1 | batch size=4 | batch size=8 | batch size=16 | batch size=32 |
|---------------------------------|--------------|--------------|--------------|---------------|---------------|
| Baseline (Pytorch)              | 5.244 ms     | 10.941 ms    | 19.585 ms    | 37.013 ms     | 68.042 ms     |
| TensorRT                        | 1.695 ms     | 4.428 ms     | 7.357 ms     | 13.899 ms     | 26.751 ms     |
| FasterTransformer (pytorch)     | 5.122 ms     | 10.796 ms    | 20.331 ms    | 37.923 ms     | 73.225 ms     |
| FasterTransformer (c++)         | 4.79 ms      | 10.73 ms     | 20.21 ms     | 37.83 ms      | 73.12 ms      |
| FasterTransformer w/ trt plugin | 4.772 ms     | 10.729 ms    | 20.211 ms    | 37.802 ms     | 73.125 ms     |

### FP16
|                                 | batch size=1 | batch size=4 | batch size=8 | batch size=16 | batch size=32 |
|---------------------------------|--------------|--------------|--------------|---------------|---------------|
| Baseline (Pytorch)              | 4.396 ms     | 4.431 ms     | 5.017 ms     | 5.442 ms      | 9.307 ms      |
| TensorRT                        | 1.133 ms     | 2.165 ms     | 3.569 ms     | 6.392 ms      | 12.478 ms     |
| FasterTransformer (Pytorch)     | 1.425 ms     | 2.089 ms     | 2.971 ms     | 4.847 ms      | 9.082 ms      |
| FasterTransformer (c++)         | 1.21 ms      | 1.93 ms      | 2.80 ms      | 4.64 ms       | 8.64 ms       |
| FasterTransformer w/ trt plugin | 1.326 ms     | 2.009 ms     | 2.900 ms     | 4.741 ms      | 8.971 ms      |

## Contents
### TensorRT (Nvidia)
- NVIDIA TensorRT is an SDK for deep learning inference. 
- TensorRT Serve the "Optimizer" and "Runtime engine".
- TensorRT has a two steps workflow: 
    - 1. Build: Build the speicific GPU optimized model for tensorRT runtime engine.
    - 2. Runtime: Run the optimized model on tensorRT runtime engine.
- Reference: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#overview

### FasterTransformer (Nvidia)
- Faster Transformer is a library that provides highly optimized transformer inference.
- FasterTransformer implements a highly optimized transformer layer for both the encoder and decoder for inference.
- Faster Transformer can optimize the transformer by model parallelism (Tensor parallelism and pipline parallelism) and fuse the operations in the transformer.
- FasterTransformer is built on top of CUDA, cuBLAS, cuBLASLt and C++ and they serve the API. 
- Reference: https://github.com/NVIDIA/FasterTransformer

### xFormers (Meta)
- xFormers is a PyTorch based library for accelerating transformer model.
- Reference: https://github.com/facebookresearch/xformers