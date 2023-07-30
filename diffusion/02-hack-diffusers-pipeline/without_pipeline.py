"""Without pipeline example
- Author: Bono (qhsh9713@gmail.com)
- Reference
    - https://towardsdatascience.com/stable-diffusion-using-hugging-face-501d8dbdd8
"""
import os
import torch
from tqdm.auto import tqdm
from diffusers import UNet2DConditionModel, LMSDiscreteScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms as tfms
from PIL import Image


# Set seed
torch.manual_seed(0)

# Set global variables
text_prompt = ["A cat in brown walkers gracefully sipping coffee in a cafe."]
batch_size = len(text_prompt)
g = 7.5
steps = 50
dim = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise RuntimeError("CUDA device not available.")

# Load clip (tokenizer, text encodoer)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

# Load Unet and scheduler
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
)
scheduler.set_timesteps(steps)
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet"
).to(device)

# Load VAE
vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae"
).to(device)


def text_encoding(prompts, maxlen=None, device=None):
    """
    A function to take a texual promt and convert it into embeddings
    """
    if maxlen is None:
        maxlen = tokenizer.model_max_length
    token = tokenizer(
        prompts,
        padding="max_length",
        max_length=maxlen,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(token.input_ids.to(device))[0]


# Encode text
text_embedding = text_encoding(prompts=text_prompt, device=device)

# Define latent noise (gaussian noise)
latents = torch.randn((batch_size, unet.in_channels, dim // 8, dim // 8))
latents = latents.to(device) * scheduler.init_noise_sigma

# Iterating through defined steps
## Adding an unconditional prompt , helps in the generation process
uncond = text_encoding(
    prompts=[""] * batch_size, maxlen=text_embedding.shape[1], device=device
)
emb = torch.cat([uncond, text_embedding])
for i, timestep in enumerate(tqdm(scheduler.timesteps)):
    # We need to scale the i/p latents to match the variance
    inp = scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
    with torch.no_grad():
        u, t = unet(inp, timestep, encoder_hidden_states=text_embedding).sample.chunk(2)

    pred = u + g * (t - u)
    latents = scheduler.step(pred, timestep, latents).prev_sample

    # Latent to image
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    pil_images[0].save(f"outputs/image_at_step_{i}.png")
