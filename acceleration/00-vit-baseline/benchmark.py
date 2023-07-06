""" Vit inference benchmark code.
- Author: Bono (qhsh9713@gmail.com)
"""
import timm
import torch
import time

# Define global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("This code only supports GPU.")
    exit(-1)
torch.cuda.synchronize()

# Define pretrained vit model
model = timm.create_model("vit_base_patch16_224", pretrained=True)
model = model.to(device)

# Define input
input = torch.randn(1, 3, 224, 224)

# Warm up
for _ in range(10):
    model(input.to(device))

# Inference
inference_times = []
start_time, end_time = (
    torch.cuda.Event(enable_timing=True),
    torch.cuda.Event(enable_timing=True),
)
with torch.no_grad():
    for _ in range(100):
        start_time.record()
        model(input.to(device))
        end_time.record()
        inference_time = start_time.elapsed_time(end_time) / 1000
        inference_times.append(inference_time)

print(
    "Vit Average Inference Time(ms) : ",
    (sum(inference_times) / len(inference_times)) * 1000,
)
