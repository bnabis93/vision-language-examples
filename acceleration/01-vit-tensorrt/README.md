## Pre-requisite
- Docker

## Export trt model
```
make convert-trt
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
- Batch size=1: `1.69581 ms (10 iterations)`
- Batch size=2: `1.71159 ms ms (10 iterations)`
- Batch size=4: `1.67822 ms (10 iterations)`
- Batch size=8: `1.69185 ms (10 iterations)`
- Batch size=16: `1.70547 ms (10 iterations)`
- Batch size=32: `1.69955 ms (10 iterations)`
- Batch size=64: `1.75058 ms (10 iterations)`
