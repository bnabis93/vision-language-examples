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

## Results
### FP32
- Batch size=1: `mean = 1.69581 ms (10 iterations)`
- Batch size=2: `mean = 2.72586 ms (10 iterations)`
- Batch size=4: `mean = 4.42871 ms (10 iterations)`
- Batch size=8: `mean = 7.35761 ms (10 iterations)`
- Batch size=16: `mean = 13.8995 ms (10 iterations)`
- Batch size=32: `mean = 26.7512 ms (10 iterations)`
- Batch size=64: `mean = 52.3231 ms (10 iterations)`

### FP16