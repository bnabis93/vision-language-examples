import clip
import torch
import torchvision.transforms as T
from PIL import Image
from segment_anything import sam_model_registry

# Load image
image = Image.open("samples/image.jpeg")

# Load CLIP and CLIP Preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
clip, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_image = clip_preprocess().unsqueeze(0).to(device)

# Load Dino v2 and Dino v2 Preprocess
dinov2_transforms = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
dinov2_image = dinov2_transforms(image)[:3].unsqueeze(0)
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

# Load SAM and SAM Preprocess
sam = sam_model_registry["vit-b"](checkpoint="./sam_vit_b_01ec64.pth")
sam_encoder = sam.image_encoder

# Get image embedding
with torch.no_grad():
    clip_image_embedding = clip.encode_image(clip_image)
    dino_image_embedding = dinov2_vitb14(dinov2_image)[0]
    sam_encoder()
