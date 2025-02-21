import torch
import os
from huggingface_hub import HfApi
from rembg import remove
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from controlnet_aux import LineartDetector
import base64
import io
from flask import jsonify
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

def generate_image_with_product(prompt:str,image:Image=None,image_url:str=None):

    checkpoint = "ControlNet-1-1-preview/control_v11p_sd15_lineart"
    
    if image is None and image_url is None:
        return jsonify({"type": "text", "data": "Product Image not found"})

    try:
        image = image if image is not None else load_image(image_url)
    except Exception as error:
        return jsonify({"type": "text", "data": f"Image not found with given URL {image_url}"})
    image= remove(image)
    image = image.resize((512, 512))

    processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

    control_image = processor(image)
    control_image.save("./images/control.png")

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)
    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    img_byte_array.seek(0)
    image_base64 = base64.b64encode(img_byte_array.read()).decode('utf-8')
    return jsonify({"type": "base64", "data": image_base64})
