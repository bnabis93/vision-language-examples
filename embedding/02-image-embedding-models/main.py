import clip
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import sys

sys.path.append("./ImageBind")
import data
from models import imagebind_model
from models.imagebind_model import ModalityType


# Load image
image_path = "samples/image.jpeg"
image = Image.open(image_path)

# Load CLIP and CLIP Preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
clip, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_image = clip_preprocess(image).unsqueeze(0).to(device)

# Load Dino v2 and Dino v2 Preprocess
dinov2_transforms = T.Compose(
    [
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
dinov2_image = dinov2_transforms(image)[:3].unsqueeze(0)
dinov2_vitb14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

# Imagebind model
imagebind_model = imagebind_model.imagebind_huge(pretrained=True)
imagebind_model.eval()
imagebind_model.to(device)
input = {ModalityType.VISION: data.load_and_transform_vision_data([image_path], device)}


# Get image embedding
with torch.no_grad():
    clip_image_embedding = clip.encode_image(clip_image)[0]
    dino_image_embedding = dinov2_vitb14(dinov2_image)[0]
    imagebind_image_embedding = imagebind_model(input)["vision"][0]

    print(clip_image_embedding.shape)
    print(dino_image_embedding.shape)
    print(imagebind_image_embedding.shape)
