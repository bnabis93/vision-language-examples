# Hack the stable-diffusion pipeline
I will divide the pipeline of stable diffsion to each components.

# Stable diffusion
- Stable Diffusion is a text-to-image `latent diffusion model` created by the researchers and engineers from CompVis, Stability AI and LAION. 
- It is trained on 512x512 images from a subset of the LAION-5B database. LAION-5B is the largest, freely accessible multi-modal dataset that currently exists.
- Paper: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752.pdf)

## Diffusion model and Stable Diffusion model
- Diffusion models are machine learning systems that are `trained to denoise random Gaussian noise step by step.`
- Diffusion models is that the reverse `denoising process is slow` because of its repeated, sequential nature. In addition, these models `consume a lot of memory` they operate in pixel space. (Pixel space is high-dimensional space)
- Stable diffusion is a kind of `latent diffusion model.` 
- Latent diffusion model can reduce the memory and compute complexity.
- The latent diffusion model's diffusion process proceeded in lower dimensions space(== latent space), not pixel space to optimize high-cost resource requirements.

# Stable Diffusion Architecture
- Model: CLIP / UNET / VAE
- The other: Scheduler / Latent Vector
![stable-diffusion-architecture](./images/stable_diffusion.png)

## The role of model.
- StableDiffusionPipeline inherits the `FromSingleFileMixin` for load checkpoint each model(clip, vae, unet).
- In the FromSingleFileMixin, the `download_from_original_stable_diffusion_ckpt` function is defined.
- We can check the default model and checkpoint reference in the `download_from_original_stable_diffusion_ckpt` function.
    - https://github.dev/huggingface/diffusers/blob/ef9824f9f79dbe0096df6797af274b316f1c4970/src/diffusers/loaders.py#L1455
    - https://github.com/huggingface/diffusers/blob/ef9824f9f79dbe0096df6797af274b316f1c4970/src/diffusers/pipelines/stable_diffusion/convert_from_ckpt.py#L1096
### CLIP
- The CLIP model consists of a tokenizer for converting text to token and a text encoder for compressing the token information.
- CLIP is used to give the text condition to the unet's embedding space. 
```
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14") if tokenizer is None else tokenizer
# text encoder also used openai/clip-vit-large-patch14 model.
```

### Unet
- Unet has a 2 inputs that `Noisy latent(gaussian noise)` and `Text embedding(CLIP's text encoder)`.
- The output is `predicted noise residual`.

### VAE(Variational Auto Encoder)
- Encoder, VAE is used to compress the image information to a lower dimension.
- Decoder, VAE is used to generate images from Unet's embedding space(encoder).
- To be precise, it is responsible for restoring the embedded image to the original image. 
- VAE is to reduce the computational time to generate High-resolution images.

## Pipeline
- A pipeline is an end-to-end class that provides a quick and easy way to use a diffusion system for inference by `bundling` independently `trained models` and `schedulers` together. 

### DiffusionPipeline
- All pipeline types inherit from the base `DiffusionPipeline` class.
    - https://github.com/huggingface/diffusers/blob/v0.19.0/src/diffusers/pipelines/pipeline_utils.py#L476
- Basically, the pipeline has a parameter named `ConfigMixin`. This is config class.
    - https://huggingface.co/docs/diffusers/v0.19.0/en/api/configuration#diffusers.ConfigMixin
    - https://github.com/huggingface/diffusers/blob/v0.19.0/src/diffusers/configuration_utils.py#L82

### StableDiffusionPipeline
```
# https://github.dev/huggingface/diffusers/blob/965e52ce611108559a0ebab75c8b421d1229c5ab/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L72
class StableDiffusionPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin):
    ...
    ...
    ...

```

## How to run?
### Set the environment
```
make env
conda activate 02-hack-diffusers-pipeline
make setup
```


## Reference
- https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img
- https://huggingface.co/blog/stable_diffusion