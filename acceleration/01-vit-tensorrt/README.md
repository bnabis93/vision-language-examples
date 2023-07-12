## Pre-requisite
- Docker

## Export trt model
### FP32 model
```
make convert-trt
...
...
[07/12/2023-13:33:21] [I] Total Host Walltime: 3.00537 s
[07/12/2023-13:33:21] [I] Total GPU Compute Time: 2.98397 s
[07/12/2023-13:33:21] [W] * GPU compute time is unstable, with coefficient of variance = 6.08549%.
[07/12/2023-13:33:21] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/12/2023-13:33:21] [I] Explanations of the performance metrics are printed in the verbose logs.
[07/12/2023-13:33:21] [I]
&&&& PASSED TensorRT.trtexec [TensorRT v8601] # trtexec --onnx=./output/model.onnx --minShapes=input0:1x3x224x224 --optShapes=input0:1x3x224x224 --maxShapes=input0:64x3x224x224 --explicitBatch --saveEngine=./output/model.plan
...
```
### FP16 model
```
make convert-fp16-trt
...
...
[07/12/2023-17:46:54] [I] === Performance summary ===
[07/12/2023-17:46:54] [I] Throughput: 912.045 qps
[07/12/2023-17:46:54] [I] Latency: min = 1.12524 ms, max = 1.15894 ms, mean = 1.13519 ms, median = 1.13501 ms, percentile(90%) = 1.14008 ms, percentile(95%) = 1.14148 ms, percentile(99%) = 1.14401 ms
[07/12/2023-17:46:54] [I] Enqueue Time: min = 0.289795 ms, max = 1.78421 ms, mean = 0.336158 ms, median = 0.314362 ms, percentile(90%) = 0.391113 ms, percentile(95%) = 0.4104 ms, percentile(99%) = 0.464111 ms
[07/12/2023-17:46:54] [I] H2D Latency: min = 0.032959 ms, max = 0.0605469 ms, mean = 0.0384768 ms, median = 0.0384521 ms, percentile(90%) = 0.0427856 ms, percentile(95%) = 0.0439148 ms, percentile(99%) = 0.04599 ms
[07/12/2023-17:46:54] [I] GPU Compute Time: min = 1.0824 ms, max = 1.09363 ms, mean = 1.08753 ms, median = 1.08752 ms, percentile(90%) = 1.0896 ms, percentile(95%) = 1.0896 ms, percentile(99%) = 1.09155 ms
[07/12/2023-17:46:54] [I] D2H Latency: min = 0.00634766 ms, max = 0.0114746 ms, mean = 0.00917772 ms, median = 0.00921631 ms, percentile(90%) = 0.010498 ms, percentile(95%) = 0.0107422 ms, percentile(99%) = 0.0112305 ms
[07/12/2023-17:46:54] [I] Total Host Walltime: 3.00314 s
[07/12/2023-17:46:54] [I] Total GPU Compute Time: 2.97874 s
[07/12/2023-17:46:54] [I] Explanations of the performance metrics are printed in the verbose logs.
[07/12/2023-17:46:54] [I]
&&&& PASSED TensorRT.trtexec [TensorRT v8601] # trtexec --onnx=./output/model.onnx --minShapes=input0:1x3x224x224 --optShapes=input0:1x3x224x224 --maxShapes=input0:64x3x224x224 --explicitBatch --fp16 --saveEngine=./output/model_fp16.plan
```


## If you want to check onnx input output name.
```
pip install onnx
python check_onnx.py

# Check_onnx.py (For get input and output name)
Inputs:  ['input0']
Outputs:  ['output0']
```


## Benchmark
- GPU: A100
- Mean inference time: 1.63666 ms

### FP32
```
make benchmark
...
...
...
[07/06/2023-10:03:30] [I] Average on 10 runs - GPU latency: 1.64355 ms - Host latency: 1.68679 ms (enqueue 0.366797 ms)
[07/06/2023-10:03:30] [I] Average on 10 runs - GPU latency: 1.64482 ms - Host latency: 1.68799 ms (enqueue 0.373926 ms)
[07/06/2023-10:03:30] [I] Average on 10 runs - GPU latency: 1.64468 ms - Host latency: 1.68721 ms (enqueue 0.36958 ms)
[07/06/2023-10:03:30] [I] Average on 10 runs - GPU latency: 1.64568 ms - Host latency: 1.6886 ms (enqueue 0.384326 ms)
[07/06/2023-10:03:30] [I] Average on 10 runs - GPU latency: 1.64487 ms - Host latency: 1.6905 ms (enqueue 0.371313 ms)
[07/06/2023-10:03:30] [I] Average on 10 runs - GPU latency: 1.64468 ms - Host latency: 1.68716 ms (enqueue 0.378491 ms)
[07/06/2023-10:03:30] [I] Average on 10 runs - GPU latency: 1.64382 ms - Host latency: 1.68679 ms (enqueue 0.395508 ms)
...
...
...
```

### FP16
```
make benchmark-fp16
...
...
...
[07/06/2023-10:03:30] [I] Average on 10 runs - GPU latency: 1.64355 ms - Host latency: 1.68679 ms (enqueue 0.366797 ms)
[07/06/2023-10:03:30] [I] Average on 10 runs - GPU latency: 1.64482 ms - Host latency: 1.68799 ms (enqueue 0.373926 ms)
[07/06/2023-10:03:30] [I] Average on 10 runs - GPU latency: 1.64468 ms - Host latency: 1.68721 ms (enqueue 0.36958 ms)
[07/06/2023-10:03:30] [I] Average on 10 runs - GPU latency: 1.64568 ms - Host latency: 1.6886 ms (enqueue 0.384326 ms)
[07/06/2023-10:03:30] [I] Average on 10 runs - GPU latency: 1.64487 ms - Host latency: 1.6905 ms (enqueue 0.371313 ms)
[07/06/2023-10:03:30] [I] Average on 10 runs - GPU latency: 1.64468 ms - Host latency: 1.68716 ms (enqueue 0.378491 ms)
[07/06/2023-10:03:30] [I] Average on 10 runs - GPU latency: 1.64382 ms - Host latency: 1.68679 ms (enqueue 0.395508 ms)
...
...
...
```

## Results
- Latency
### FP32
- Batch size=1: `mean = 1.69581 ms (10 iterations)`
- Batch size=2: `mean = 2.72586 ms (10 iterations)`
- Batch size=4: `mean = 4.42871 ms (10 iterations)`
- Batch size=8: `mean = 7.35761 ms (10 iterations)`
- Batch size=16: `mean = 13.8995 ms (10 iterations)`
- Batch size=32: `mean = 26.7512 ms (10 iterations)`
- Batch size=64: `mean = 52.3231 ms (10 iterations)`

### FP16
- Batch size=1: `mean = 1.13391 ms (10 iterations)`
- Batch size=2: `mean = 1.54519 ms (10 iterations)`
- Batch size=4: `mean = 2.16532 ms (10 iterations)`
- Batch size=8: `mean = 3.56952 ms (10 iterations)`
- Batch size=16: `mean = 6.39221 ms (10 iterations)`
- Batch size=32: `mean = 12.4785 ms (10 iterations)`
- Batch size=64: `mean = 24.2022 ms (10 iterations)`
- Batch size=256: `mean = 88.7347 ms (10 iterations)`
- Batch size=512: `mean = 176.361 ms (10 iterations)`