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

### Run ViT on C++
- Run ViT on binary file.
- Firstly we use ./bin/vit_gemm as the tool to search the best GEMM configuration. And then run ./bin/vit_example
```
# FP32 mode.
export NVIDIA_TF32_OVERRIDE=0

## Find best gemm config
./bin/vit_gemm <batch_size> <img_size> <patch_size> <embed_dim> <head_number> <with_cls_token> <is_fp16> <int8_mode> 
./bin/vit_gemm 1 224 16 768 12 1 0 0

## ViT
./bin/vit_example <batch_size> <img_size> <patch_size> <embed_dim> <head_number> <layer_num> <with_cls_token> <is_fp16>
./bin/vit_example 1 224 16 768 12 12 1 0

# FP16
./bin/vit_gemm 1 224 16 768 12 1 1 0
./bin/vit_example 1 224 16 768 12 12 1 1
```

### Run ViT on Pytorch
```
cd $WORKSPACE/examples/pytorch/vit/ViT-quantization
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16.npz
pip install ml_collections

cd $WORKSPACE/examples/pytorch/vit
export NVIDIA_TF32_OVERRIDE=0

python infer_visiontransformer_op.py \
  --model_type=ViT-B_16  \
  --img_size=224 \
  --pretrained_dir=./ViT-quantization/ViT-B_16.npz \
  --batch-size=1 \
  --th-path=$WORKSPACE/build/lib/libth_transformer.so
```


### TensorRT Plugin
```
# Get ViT weight file
cd $WORKSPACE/examples/pytorch/vit/ViT-quantization
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16.npz
pip install ml_collections

#FP32 engine build & infer
cd $WORKSPACE/examples/tensorrt/vit
export NVIDIA_TF32_OVERRIDE=0
python infer_visiontransformer_plugin.py \
  --model_type=ViT-B_16 \
  --img_size=224 \
  --pretrained_dir=$WORKSPACE/examples/pytorch/vit/ViT-quantization/ViT-B_16.npz \
  --plugin_path=../../../build/lib/libvit_plugin.so \
  --batch-size=8

#FP16 engine build & infer
python infer_visiontransformer_plugin.py \
  --model_type=ViT-B_16 \
  --img_size=224 \
  --pretrained_dir=$WORKSPACE/examples/pytorch/vit/ViT-quantization/ViT-B_16.npz \
  --plugin_path=../../../build/lib/libvit_plugin.so \
  --batch-size=1 \
  --fp16
```


## Experiment
### Server
- CPU: AMD EPYC Processor (with IBPB)
- GPU: A100 x1

## Results
### Run ViT on C++
FP32
- Batch size=1: `FT-CPP-time 4.79 ms (100 iterations)`
- Batch size=4: `FT-CPP-time 10.73 ms (100 iterations)`
- Batch size=8: `FT-CPP-time 20.21 ms (100 iterations)`
- Batch size=16: `FT-CPP-time 37.83 ms (100 iterations)`
- Batch size=32: `FT-CPP-time 73.12 ms (100 iterations)`

FP16
- Batch size=1: `FT-CPP-time 1.21 ms (100 iterations)`
- Batch size=4: `FT-CPP-time 1.93 ms (100 iterations)`
- Batch size=8: `FT-CPP-time 2.80 ms (100 iterations)`
- Batch size=16: `FT-CPP-time 4.64 ms (100 iterations)`
- Batch size=32: `FT-CPP-time 8.64 ms (100 iterations)`

### Run ViT on pytorch op
FP32
- Batch size=1: `5.122 ms`
- Batch size=4: `10.796 ms`
- Batch size=8: `20.331 ms`
- Batch size=16: `37.923 ms`
- Batch size=32: `73.225 ms`

FP16
- Batch size=1: `1.425 ms`
- Batch size=4: `2.089 ms`
- Batch size=8: `2.971 ms`
- Batch size=16: `4.847 ms`
- Batch size=32: `9.082 ms`

### TensorRT plugin
- fastertransformer speed slower than pytorch : https://github.com/NVIDIA/FasterTransformer/issues/325
- [Recommendation] Use fp16 / bf16 model than fp32.
- There's almost no difference.

FP32
- Batch size=1
    - plugin time :  4.7725653648376465 ms
    - torch time :  5.4483866691589355 ms
- Batch size=4
    - plugin time :  10.72922945022583 ms
    - torch time :  11.971831321716309 ms
- Batch size=8
    - plugin time :  20.211186408996582 ms
    - torch time :  20.98031759262085 ms
- Batch size=16
    - plugin time :  37.80216932296753 ms
    - torch time :  40.00376224517822 ms
- Batch size=32
    - plugin time :  73.12519073486328 ms
    - torch time :  73.00717830657959 ms

FP16
- Batch size=1
    - plugin time :  1.3268542289733887 ms
    - torch time :  5.363030433654785 ms
- Batch size=4
    - plugin time :  2.0092082023620605 ms
    - torch time :  5.434269905090332 ms
- Batch size=8
    - plugin time :  2.9003071784973145 ms
    - torch time :  5.36419153213501 ms
- Batch size=16
    - plugin time :  4.741630554199219 ms
    - torch time :  7.014093399047852 ms
- Batch size=32
    - plugin time :  8.971138000488281 ms
    - torch time :  12.564880847930908 ms

## Reference
- https://github.com/NVIDIA/FasterTransformer/blob/main/docs/QAList.md