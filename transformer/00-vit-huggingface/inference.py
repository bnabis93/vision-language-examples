""" Vit inference with huggingface
- Author: Bono (qhsh9713@gmail.com)
"""
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--image_path', type=str, default='samples/test01.jpeg')
args = arg_parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reference from : https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTForImageClassification
# You can also check this huggingface repo for check various vit pretrained model. https://huggingface.co/google
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.to(device)

# Load image
image = Image.open(args.image_path)

# Preprocessing
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
inputs = processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values

# Inference
with torch.no_grad():
    outputs = model(pixel_values)
logits = outputs.logits
print(f"logit shape : {logits.shape}")

predictions = logits.argmax(dim=-1)
print(f"Predictions class : {model.config.id2label[predictions.item()]}")