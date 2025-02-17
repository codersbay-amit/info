import numpy as np
import random

import spaces
from diffusers import DiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_repo_id = "stabilityai/stable-diffusion-3.5-large-turbo"

if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float32

pipe = DiffusionPipeline.from_pretrained(model_repo_id, torch_dtype=torch_dtype)
pipe.enable_model_cpu_offload()
pipe = pipe.to(device)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

@spaces.GPU
def generate(
    prompt,
    negative_prompt="",
    seed=42,
    randomize_seed=True,
    width=1024,
    height=1024,
    guidance_scale=4.5,
    num_inference_steps=4,
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator().manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
    ).images[0]

    return image, seed


"""an advertisement for Galaxy S23, close up photography of a phone, simple design, with the title ("Limited Time Offer" text logo),with the subtitle ( "" text logo),with the button_text ( "" text logo), clean design, vibrant images, product photography, white background (#ffffff)'<lora:Harrlogos_v1.1:1>"""