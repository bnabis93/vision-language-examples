import numpy as np
import torch
import clip
from PIL import Image

# Set the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Set the input images and texts
## Descriptions with particular emphasis on shape.
shape_description_data = {
    "banana": {
        "filename": "banana.jpeg",
        "description": "Banana with a curved shape.",
    },
    "duck": {
        "filename": "duck.png",
        "description": "A duck with wings and a beak.",
    },
    "duck-banana": {
        "filename": "duck-banana.png",
        "description": "A banana with a duck for a head and a curved body is stuck to a piece of tape.",
    },
    "cat": {
        "filename": "cat.png",
        "description": "A cat with a fluffy coat.",
    },
    "mountain": {
        "filename": "mountain.png",
        "description": "Pointed Mountains.",
    },
    "cat_mountain": {
        "filename": "cat_mountain.png",
        "description": "Pointed mountains and peaks have cat heads.",
    },
}

## Descriptions with particular emphasis on color.
color_description_data = {
    "banana": {
        "filename": "banana.jpeg",
        "description": "A yellow banana on a white background.",
    },
    "duck": {
        "filename": "duck.png",
        "description": "A cute duck with an orange beak and yellow feather color.",
    },
    "duck-banana": {
        "filename": "duck-banana.png",
        "description": "A duck with an orange beak for a head and a banana for a body on a white background with silver tape.",
    },
    "cat": {
        "filename": "cat.png",
        "description": "A cat with white fur, brown ears and tail.",
    },
    "mountain": {
        "filename": "mountain.png",
        "description": "Below is a snowy mountain with a forest.",
    },
    "cat_mountain": {
        "filename": "cat_mountain.png",
        "description": "Below is a lake, and at the top of the snowy mountain is a white cat's face.",
    },
}

# Images and Texts
images = []
color_texts = []
shape_texts =  []

for key, _ in color_description_data.items():
    image = Image.open(f"samples/{color_description_data[key]['filename']}").convert(
        "RGB"
    )
    text = color_description_data[key]["description"]
    images.append(preprocess(image))
    color_texts.append(text)

for key, _ in shape_description_data.items():
    text = shape_description_data[key]["description"]
    shape_texts.append(text)

image_input = torch.tensor(np.stack(images)).cuda()
color_text_tokens = clip.tokenize(["This is " + desc for desc in color_texts]).cuda()
shape_text_tokens = clip.tokenize(["This is " + desc for desc in shape_texts]).cuda()

# Encodeing
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    color_text_features = model.encode_text(color_text_tokens).float()
    shape_text_features = model.encode_text(shape_text_tokens).float()

# Calculating cosine similarity
image_features /= image_features.norm(dim=-1, keepdim=True)
color_text_features /= color_text_features.norm(dim=-1, keepdim=True)
shape_text_features /= shape_text_features.norm(dim=-1, keepdim=True)
color_similarity = color_text_features.cpu().numpy() @ image_features.cpu().numpy().T
shape_similarity = shape_text_features.cpu().numpy() @ image_features.cpu().numpy().T

# Descriptions with particular emphasis on color.
print("Descriptions with particular emphasis on color. sim : \n", color_similarity)
print("Descriptions with particular emphasis on shape. sim : \n", shape_similarity)
