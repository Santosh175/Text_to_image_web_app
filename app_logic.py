from typing import Literal
from diffusers import StableDiffusionPipeline
import torch
import time

import os
import io
import requests
from PIL import Image

from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler



seed = 2024
generator = torch.manual_seed(seed)

NUM_ITERS_TO_RUN = 1
NUM_INFERENCE_STEPS = 25
NUM_IMAGES_PER_PROMPT = 1

    
# Add your hugging face hub token here.
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "Your_hub_API_KEY"

def text2image(
    prompt: str,
    repo_id: Literal[
        "prithivMLmods/Canopus-Realism-LoRA",
        "black-forest-labs/FLUX.1-dev",
        "SG161222/RealVisXL_V4.0_Lightning",
        "prompthero/openjourney",
        "stabilityai/stable-diffusion-2-1",
        "runwayml/stable-diffusion-v1-5",
        "SG161222/RealVisXL_V3.0",
        "CompVis/stable-diffusion-v1-4",
    ],
):
    start = time.time()

    HF_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    API_URL = f"https://api-inference.huggingface.co/models/{repo_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs":prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    image_bytes = response.content
    image = Image.open(io.BytesIO(image_bytes))
    upscaled_image = image.resize((2048,2048))

    
    end = time.time()
    return upscaled_image, start, end
