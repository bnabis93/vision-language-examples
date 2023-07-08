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

## Benchmark
### Setup
```
# Create docker env
make docker-setup
export WORKSPACE=/workspace/FasterTransformer

# Install additional dependencies
cd $WORKSPACE
pip install -r examples/pytorch/vit/requirement.txt

# Build Faster Trasnformer
cd $WORKSPACE
git submodule update --init
mkdir -p build
cd build
## Note: xx is the compute capability of your GPU. For example, 60 (P40) or 61 (P4) or 70 (V100) or 75(T4) or 80 (A100).
cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_TRT=ON ..
make -j12
```

### Run
- Run ViT on binary file.
- Firstly we use ./bin/vit_gemm as the tool to search the best GEMM configuration. And then run ./bin/vit_example
- After gemm config setup, `FT-CPP-time 4.79 ms (100 iterations)`
```
# TF32 mode.
export NVIDIA_TF32_OVERRIDE=0

# Find best gemm config
./bin/vit_gemm <batch_size> <img_size> <patch_size> <embed_dim> <head_number> <with_cls_token> <is_fp16> <int8_mode> 
./bin/vit_gemm 1 224 16 768 12 1 0 0

# ViT
./bin/vit_example <batch_size> <img_size> <patch_size> <embed_dim> <head_number> <layer_num> <with_cls_token> <is_fp16>
./bin/vit_example 1 224 16 768 12 12 1 0
```


## Reference
- https://github.com/NVIDIA/FasterTransformer/blob/main/docs/QAList.md