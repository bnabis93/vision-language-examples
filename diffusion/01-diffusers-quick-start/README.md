# Diffusers

## What is the diffuser?

## Concept of diffuser's components
- Diffusion pipeline: ene-to-end diffusion pipeline for inference. It already has a pretrained diffsion model, and then it can be used for inference. Pipeline stores all components (models, schedulers, and processors). Also provides model loading, downloading and saving.
- Model: Pretrained diffusion model. It can be used for inference. Diffusers model built on the base class `ModelMixin` that is a `torch.nn.module`.
    - The model configuration is a 🧊 frozen 🧊 dictionary, which means those parameters can’t be changed after the model is created. Therefore, the model is always static and reproducible. We consider the other parameters. 
- Schedulers: In the diffusion process, the scheduler is operated. Scheduler also be called `Samplers` in other diffusion models implementations. The scheduler can control denoising steps, denoising speed, noise level, and other parameters. (quality trade-off)


## How to run?
### Set the environment
```
make env
conda activate 01-diffusers-quick-start
make setup
```

### Pipeline example
```
python pipeline.py
```

### Model and scheduler example
```
python model.py
```