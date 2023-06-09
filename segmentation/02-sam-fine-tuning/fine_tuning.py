import os
import torch
import numpy as np
import monai
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataloader import NpzDataset
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

join = os.path.join

# Set global variable
## [TODO]: Will be managed by config file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
npz_dataset_path = "data/demo2D_vit_b"
data_batch = 4
model_type = "vit_b"
checkpoint = "models/sam_vit_b_01ec64.pth"
save_path = "./outputs"
num_epochs = 100
os.makedirs(save_path, exist_ok=True)

# Set seeds
torch.manual_seed(2023)
np.random.seed(2023)

# Load the dataset
dataset = NpzDataset(npz_dataset_path)
dataloader = DataLoader(dataset, batch_size=data_batch, shuffle=True)

# Load the model and model parameters
model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
model.train()
optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

# Train the model
losses = []
best_loss = 1e10
for epoch in range(num_epochs):
    epoch_loss = 0
    ## img_embed: (B, 256, 64, 64), gt2D: (B, 1, 256, 256), bboxes: (B, 4), points: (B, 2), B= batch size
    for step, (image_embedding, gt2D, boxes, points) in enumerate(tqdm(dataloader)):
        with torch.no_grad():  # Encoder part, no need to calculate gradients.
            box_np = boxes.numpy()
            sam_trans = ResizeLongestSide(model.image_encoder.img_size)

            # Note: Points are input to the model in (x,y) format
            # Note: labels 1 (foreground point) or 0 (background point).
            point = sam_trans.apply_coords(
                points.numpy(), (gt2D.shape[-2], gt2D.shape[-1])
            )
            point = torch.as_tensor(point, dtype=torch.float, device=device)
            if len(point.shape) == 2:
                point = point[:, None, :]  # (B, 1, 4)
            label = np.ones((point.shape[0], 1))
            label = torch.as_tensor(label, dtype=torch.int, device=device)

            # get prompt embeddings
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=(point, label),
                boxes=None,
                masks=None,
            )
        # Decoder part, calculate gradients.
        mask_predictions, _ = model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        loss = seg_loss(mask_predictions, gt2D.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= step
    losses.append(epoch_loss)

    print(f"EPOCH: {epoch}, Loss: {epoch_loss}")
    # save the latest model checkpoint
    torch.save(model.state_dict(), join(save_path, "sam_model_latest.pth"))
    # save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), join(save_path, "sam_model_best.pth"))

# plot loss
plt.plot(losses)
plt.title("Dice + Cross Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(join(save_path, "train_loss.png"))
plt.close()
