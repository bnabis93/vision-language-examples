# Acceleration
Methods for transformer acceleration.

|                    | batch size=1 | batch size=4 | batch size=8 | batch size=16 | batch size=32 |
|--------------------|--------------|--------------|--------------|---------------|---------------|
| Baseline (Pytorch) | 12.456 ms    | 26.858 ms    | 47.407 ms    | 93.633 ms     | 171.224 ms    |
| TensorRT           | 1.695 ms     | 4.428 ms     | 7.357 ms     | 13.899 ms     | 26.751 ms     |
| FasterTransformer  | 4.79 ms      | 10.73 ms     | 20.21 ms     | 37.83 ms      | 73.12 ms      |
| xFormers           |              |              |              |               |               |

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