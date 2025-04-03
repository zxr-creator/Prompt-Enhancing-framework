from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import numpy as np

class ImageGenerator:
    def __init__(self, model_name="sd-legacy/stable-diffusion-v1-5", device="cpu"):
        self.device = device
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        ).to(device)

    def generate(self, init_image: Image.Image, prompt: str, strength=0.75, guidance_scale=7.5, num_inference_steps=50):
        """
        init_image: PIL.Image - input image
        prompt: str - text prompt for image generation
        strength: float - image modification strength, 0.0 means no change, 1.0 means full change (default is 0.75)
        guidance_scale: float - higher guidance scale encourages to generate images closely linked to the text prompt (default is 7.5)
        """
        init_image = init_image.convert("RGB").resize((512, 512))
        result = self.pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )
        return result.images[0]