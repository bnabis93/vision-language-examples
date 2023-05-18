import torch
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T

from torchvision.datasets import CIFAR10
from sklearn.manifold import TSNE


# Load all cifiar10 dataset with label for tsne
dinov2_transforms = T.Compose(
    [
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
cifar10 = CIFAR10(
    root="./data", download=True, train=False, transform=dinov2_transforms
)
cifar10 = torch.utils.data.Subset(cifar10, range(1000))


# Load Dino v2 and Dino v2 Preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
dinov2_vitb14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

# Get image embedding
embeddings = []
labels = []
with torch.no_grad():
    for image, label in tqdm.tqdm(cifar10):
        dinov2_image = image[:3].unsqueeze(0)
        dinov2_image_embedding = dinov2_vitb14(dinov2_image)[0]

        embeddings.append(dinov2_image_embedding.cpu().numpy())
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
plt.title("dinov2 Image Embedding")
plt.xlabel("tsne_x")
plt.ylabel("tsne_y")
plt.legend()
plt.savefig("dinov2_tsne.png")
