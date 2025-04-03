from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusion3Pipeline
import torch
from PIL import Image
import numpy as np

class ImageGenerator:
    def __init__(self, model_name="stabilityai/stable-diffusion-2-1", device="mps"):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
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

class TextToImageGenerator:
    def __init__(self, model_name="stabilityai/stable-diffusion-3.5-medium", device="cpu"):
        """
        Initialize the TextToImageGenerator with the specified model and device.
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        ).to(device)

    def generate(self, prompt: str, num_inference_steps=40, guidance_scale=4.5):
        """
        Generate an image from a text prompt.

        Args:
            prompt (str): The text prompt for image generation.
            num_inference_steps (int): Number of denoising steps (default is 40).
            guidance_scale (float): Higher guidance scale encourages images closely linked to the text prompt (default is 4.5).

        Returns:
            PIL.Image.Image: The generated image.
        """
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        return result.images[0]