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


## Reference
- https://github.com/facebookresearch/xformers
- https://facebookresearch.github.io/xformers/what_is_xformers.html
