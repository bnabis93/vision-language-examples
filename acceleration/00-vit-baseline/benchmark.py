""" Vit inference benchmark code.
- Author: Bono (qhsh9713@gmail.com)
"""
import timm
import torch
import time

# Define global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define pretrained vit model
model = timm.create_model("vit_base_patch16_224", pretrained=True)
model = model.to(device)

# Define input
input = torch.randn(1, 3, 224, 224)

# Warm up
for _ in range(10):
    model(input.to(device))

# Inference
inference_time = []
with torch.no_grad():
    for _ in range(100):
        start = time.time()
        model(input.to(device))
        end = time.time()
        inference_time.append(end - start)

print("Vit Average Inference Time: ", sum(inference_time) / len(inference_time))
