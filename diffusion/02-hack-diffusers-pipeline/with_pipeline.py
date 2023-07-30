"""Pipeline example
- Author: Bono (qhsh9713@gmail.com)
"""
import os
import torch
from diffusers import DiffusionPipeline

# Set seed
torch.manual_seed(0)

# Set the pipeline
generator = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
generator.to("cuda")
image = generator("An image of a squirrel in Picasso style").images[0]

# Save image
if not os.path.exists("outputs"):
    os.makedirs("outputs")
image.save("outputs/image_of_squirrel_painting.png")
