# Faster Transformer ViT model Benchmark

## Faster Transformer
- FasterTransformer provides flexible APIs and highly optimized kernels.
- Provide optimized models for supported models only.
- FasterTransformer provide C API, `TF/Pytorch Operation`. Also able to wrap the C++ codes to integrate FasterTransformer.
- Recommend to use `Docker image` to use faster transformer.

### Supported GPU
- Compute Compatibility >= 7.0 such as V100, T4 and A100.
- CPU spec also has an impact.
> FasterTransformer’s approach is to offload the computational workloads to GPUs with the memory operations overlapped with them. So FasterTransformer performance is mainly decided by what kinds of GPUs and I/O devices are used. However, when the batch size and sequence length are both small, kernel launching is the bottleneck and hence worse CPU may lead to worse performance.

## If you use pytorch...
- Load the checkpoint and put the weight tensor into FasterTransformer directly.
> users can load the checkpoint and put the weight tensor into FasterTransformer directly. Users can also load the model in other formats, like numpy, and put them into FasterTransformer directly like the weight tensor.

## Reference
- https://github.com/NVIDIA/FasterTransformer/blob/main/docs/QAList.md