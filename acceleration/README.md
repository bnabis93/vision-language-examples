# Acceleration
Methods for transformer acceleration.

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