"""Benchmark ViT trt model.

- Author: Bono (bnabis93, github)
- Email: qhsh9713@gmail.com
"""
import argparse
import time
import torch
from trt.trt_infer import TrtModel

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument(
    "--trt_path",
    default="./output/model.trt",
    type=str,
    help="Pretrained model weight path",
)
args = parser.parse_args()


def main():
    """Main function for benchmarking."""
    # Define trt model
    trt_engine_path = args.trt_path
    trt_net = TrtModel(trt_engine_path)
    inputs = torch.randn(1, 3, 224, 224)

    # Benchmark
    inference_times = []
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        trt_net(inputs)
        end = time.time()
        torch.cuda.synchronize()
        inference_times.append((end - start) * 1000)

    print(
        f"ViT TensorRT average inference time : {sum(inference_times)/len(inference_times)}ms"
    )


if __name__ == "__main__":
    main()
