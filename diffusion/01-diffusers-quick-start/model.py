import os
import torch
import tqdm
import PIL.Image
import numpy as np
from diffusers import UNet2DModel, DDPMScheduler


def save_sample(sample, step, save_dir="outputs"):
    """
    Save the sample image.
    """
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    print(f"Image at step {step}")

    # Save image
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_pil.save(os.path.join(save_dir, f"image_at_step_{step}.png"))


# Set seed
torch.manual_seed(0)

# Load model
repo_id = "google/ddpm-cat-256"
model = UNet2DModel.from_pretrained(repo_id)
print("Model config: \n", model.config)

# Set the scheduler
scheduler = DDPMScheduler.from_config(repo_id)
print("Scheduler config: \n", scheduler)

# Generate a sample (Gaussian noise)
noisy_sample = torch.randn(
    1, model.config.in_channels, model.config.sample_size, model.config.sample_size
)
print("Sample size: ", noisy_sample.shape)

# Inference
model.to("cuda")
noisy_sample = noisy_sample.to("cuda")
sample = noisy_sample
for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    with torch.no_grad():
        residual = model(sample, t).sample

    # 2. compute less noisy image and set x_t -> x_t-1, update sample
    sample = scheduler.step(residual, t, sample).prev_sample

    # 3. optionally look at image
    if (i + 1) % 50 == 0:
        save_sample(sample, i + 1)
