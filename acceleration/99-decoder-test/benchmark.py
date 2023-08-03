""" Vit inference benchmark code.
- Author: Bono (qhsh9713@gmail.com)
"""
import torch
import time
from decoder import Decoder
from torch.autograd import Variable

# Define global variables
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("This code only supports GPU.")
    exit(-1)

decoder = Decoder(num_classes=10).to(device)
x = Variable(torch.randn(1, 256, 32, 32)).to(device)
# Warm up
for _ in range(10):
    decoder(x)

# Inference
inference_times = []
with torch.no_grad():
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()
        decoder(x)
        torch.cuda.synchronize()
        end = time.time()
        inference_times.append((end - start) * 1000)

print(f"decoder average inference time : {sum(inference_times)/len(inference_times)}ms")
