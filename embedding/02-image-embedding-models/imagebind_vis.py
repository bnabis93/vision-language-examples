import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10
from sklearn.manifold import TSNE
import tqdm
import sys
import torchvision.transforms as T

sys.path.append("./ImageBind")
from models import imagebind_model
from models.imagebind_model import ModalityType

# Load all cifiar10 dataset with label for tsne
device = "cuda" if torch.cuda.is_available() else "cpu"
data_transform = T.Compose(
    [
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)
cifar10 = CIFAR10(root="./data", download=True, train=False, transform=data_transform)

# Imagebind model
imagebind_model = imagebind_model.imagebind_huge(pretrained=True)
imagebind_model.eval()
imagebind_model.to(device)

# Get image embedding
embeddings = []
labels = []
with torch.no_grad():
    for image, label in tqdm.tqdm(cifar10):
        input = {ModalityType.VISION: image.unsqueeze(0).to(device)}
        imagebind_image_embedding = imagebind_model(input)["vision"][0]
        embeddings.append(imagebind_image_embedding.cpu().numpy())
        labels.append(label)

# Get tsne
tsne = TSNE(n_components=2, random_state=0)
tsne_embeddings = np.array(tsne.fit_transform(np.array(embeddings)))
labels = np.array(labels)

# Plot tsne with label
plt.figure(figsize=(12, 12))
cifar_class = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
for i, label in zip(range(10), cifar_class):
    idx = np.where(labels == i)
    plt.scatter(
        tsne_embeddings[idx, 0], tsne_embeddings[idx, 1], marker=".", label=label
    )

# Save the plot
plt.title("Imagebind Image Embedding")
plt.xlabel("tsne_x")
plt.ylabel("tsne_y")
plt.legend()
plt.savefig("imagebind_tsne.png")
