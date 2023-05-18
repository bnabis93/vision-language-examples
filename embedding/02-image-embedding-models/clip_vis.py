import clip
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10
from sklearn.manifold import TSNE
import tqdm

# Load all cifiar10 dataset with label for tsne
cifar10 = CIFAR10(root="./data", download=True, train=False)
cifar10 = torch.utils.data.Subset(cifar10, range(1000))

# Load CLIP and CLIP Preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
clip, clip_preprocess = clip.load("ViT-B/32", device=device)

# Get image embedding
embeddings = []
labels = []
with torch.no_grad():
    for image, label in tqdm.tqdm(cifar10):
        clip_image = clip_preprocess(image).unsqueeze(0).to(device)
        clip_image_embedding = clip.encode_image(clip_image)[0]
        embeddings.append(clip_image_embedding.cpu().numpy())
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
plt.title("CLIP Image Embedding")
plt.xlabel("tsne_x")
plt.ylabel("tsne_y")
plt.legend()
plt.savefig("clip_tsne.png")
