import numpy as np
import monai
import torch
import matplotlib.pyplot as plt

from data_loader import SAMDataset

from torch.utils.data import DataLoader
from statistics import mean
from PIL import Image
from datasets import load_dataset
from transformers import SamModel, SamProcessor
from torch.optim import Adam
from tqdm import tqdm


# [TODO] Hyperparameter will be replaced with a better one (like omegaconf)
num_epochs = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load datasets from huggingface
dataset = load_dataset("nielsr/breast-cancer", split="train")
print(dataset)

# Visualize datasets
example_image = dataset[0]["image"]
example_mask = dataset[0]["label"]
example_mask = Image.fromarray(np.array(example_mask) * 255).convert("L") # Single channel
example_image.save("outputs/example_image.png")
example_mask.save("outputs/example_mask.png")

# Create pytorch dataset for fine-tuning
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
train_dataset = SAMDataset(dataset=dataset, processor=processor)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True) # Convert to torch data loader format

# Load model
model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

# Train the model
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
model.to(device)
model.train()

mean_losses = []
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
        # forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_points=batch["input_points"].to(device),
                        multimask_output=False)

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        # backward pass (compute gradients of parameters w.r.t. loss)
        optimizer.zero_grad()
        loss.backward()

        # optimize
        optimizer.step()
        epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')
    mean_losses.append(mean(epoch_losses))


# Save the model
model.save_pretrained("./sam_breast_cancer")

# Save the loss in figure
plt.plot(mean_losses)
plt.savefig("outputs/loss.png")