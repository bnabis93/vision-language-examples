""" Vit inference benchmark code.
- Author: Bono (qhsh9713@gmail.com)
"""
import torch
import time
from unet import Encoder, Decoder

# Define global variables
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("This code only supports GPU.")
    exit(-1)

# Define encoder and decoder
encoder = Encoder()
encoder = encoder.to(device)
encoder_input = torch.randn(1, 3, 572, 572)
embedding = Encoder(encoder_input.to(device))


decoder = Decoder()
decoder = decoder.to(device)

# Define input
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
for batch in batch_sizes:
    input = torch.randn(1, 512, 28, 28)

    # Warm up
    for _ in range(10):
        decoder(input.to(device), embedding[::-1][1:])

    # Inference
    inference_times = []
    with torch.no_grad():
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.time()
            decoder(input.to(device), embedding[::-1][1:])
            torch.cuda.synchronize()
            end = time.time()
            inference_times.append((end - start) * 1000)

    print(
        f"batch size :{batch}\nViT average inference time : {sum(inference_times)/len(inference_times)}ms"
    )
