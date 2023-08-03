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

# Define pretrained vit model
model = Decoder()
model = model.to(device)

# Define input
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
for batch in batch_sizes:
    input = torch.randn(1, 512, 28, 28)

    # Warm up
    for _ in range(10):
        model(input.to(device))

    # Inference
    inference_times = []
    with torch.no_grad():
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.time()
            model(input.to(device))
            torch.cuda.synchronize()
            end = time.time()
            inference_times.append((end - start) * 1000)

    print(
        f"batch size :{batch}\nViT average inference time : {sum(inference_times)/len(inference_times)}ms"
    )
