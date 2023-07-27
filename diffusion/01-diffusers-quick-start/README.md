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
Loading pipeline components...:  29%|██████████████████████████████████████████                                                                                                         | 2/7 [00:00<00:01,  3.08it/s]`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["id2label"]` will be overriden.
`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["bos_token_id"]` will be overriden.
`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["eos_token_id"]` will be overriden.
Loading pipeline components...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  4.57it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:09<00:00,  5.32it/s]

ls outputs
image_of_squirrel_painting.png
```

### Model and scheduler example
```
python model.py
...
...
...
Image at step 900
 95%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏        | 949/1000 [00:31<00:01, 36.55it/s]Image at step 950
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍| 997/1000 [00:32<00:00, 36.61it/s]Image at step 1000
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:33<00:00, 30.21it/s]

ls outputs
ls
image_at_step_100.png   image_at_step_200.png  image_at_step_350.png  image_at_step_50.png   image_at_step_600.png  image_at_step_750.png  image_at_step_900.png
image_at_step_1000.png  image_at_step_250.png  image_at_step_400.png  image_at_step_500.png  image_at_step_650.png  image_at_step_800.png  image_at_step_950.png
image_at_step_150.png   image_at_step_300.png  image_at_step_450.png  image_at_step_550.png  image_at_step_700.png  image_at_step_850.png
```
