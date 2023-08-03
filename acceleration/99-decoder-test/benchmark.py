""" Vit inference benchmark code.
- Author: Bono (qhsh9713@gmail.com)
"""
import torch
import time
from decoder import Decoder

# Define global variables
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("This code only supports GPU.")
    exit(-1)

# Define encoder and decoder
in_channels_list = [
    256,
    128,
    64,
    32,
]  # For example, matching the output channels of encoder's stages
out_channels_list = [
    128,
    64,
    32,
    16,
]  # You can modify these numbers based on your model architecture.

decoder = Decoder(in_channels_list, out_channels_list)
decoder = decoder.to(device)

input_tensor = torch.randn(1, 32, 16, 16)
skip1 = torch.randn(1, 256, 64, 64)
skip2 = torch.randn(1, 128, 32, 32)
skip3 = torch.randn(1, 64, 64, 64)
skip4 = torch.randn(1, 32, 128, 128)
skip_connections = [skip1, skip2, skip3, skip4]

# Define input
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
for batch in batch_sizes:
    input = torch.randn(1, 512, 28, 28)

    # Warm up
    for _ in range(10):
        decoder(input_tensor, skip_connections)

    # Inference
    inference_times = []
    with torch.no_grad():
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.time()
            decoder(input_tensor, skip_connections)
            torch.cuda.synchronize()
            end = time.time()
            inference_times.append((end - start) * 1000)

    print(
        f"batch size :{batch}\nViT average inference time : {sum(inference_times)/len(inference_times)}ms"
    )
