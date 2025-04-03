import pandas as pd
from ImageGenerator import ImageGenerator, TextToImageGenerator  # Ensure this is imported from your ImageGenerator module
from PIL import Image
import requests
from io import BytesIO
from huggingface_hub import login  # Import Hugging Face login
import yaml

# Hugging Face authentication
def authenticate_huggingface_from_yaml(file_path: str):
    """
    Authenticate with Hugging Face using a personal access token from a YAML file.
    """
    try:
        with open(file_path, "r") as file:
            keys = yaml.safe_load(file)  # Load YAML content
            token = keys.get("huggingface", {}).get("token")  # Get Hugging Face token
            if not token:
                raise ValueError("Hugging Face token not found in the YAML file.")
            login(token)
            print("Successfully authenticated with Hugging Face.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error during authentication: {e}")

def process_prompts(start_row, end_row, mode="img2img"):
    """
    Process prompts and generate images.

    Args:
        start_row (int): Start row index in the CSV file.
        end_row (int): End row index in the CSV file.
        mode (str): Mode of generation, either "img2img" or "text2img".
    """
    # Load prompts and image URLs from CSV
    data = pd.read_csv("../data/diffusion_prompts.csv")
    selected_data = data.iloc[start_row:end_row]  # Select specific rows

    # Preload images for img2img mode
    images = {}
    if mode == "img2img":
        for index, row in selected_data.iterrows():
            image_url = row['url']
            response = requests.get(image_url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                images[index] = img
                print(f"Preloaded image for index {index}")
            else:
                print(f"Failed to download image from {image_url}")
                images[index] = None

    # Process each prompt
    for index, row in selected_data.iterrows():
        prompt = row['prompt']

        if mode == "img2img":
            img = images.get(index)
            if img is None:
                print(f"Skipping index {index} due to missing image.")
                continue

            # Generate new image based on the prompt
            gen = ImageGenerator()
            new_img = gen.generate(img, prompt=prompt)

            # Save the generated image
            output_path = f"output_img2img_{index}.jpg"
            new_img.save(output_path)
            print(f"Saved generated image to {output_path}")

        elif mode == "text2img":
            # Generate image directly from text prompt
            gen = TextToImageGenerator()
            new_img = gen.generate(prompt=prompt)

            # Save the generated image
            output_path = f"output_text2img_{index}.jpg"
            new_img.save(output_path)
            print(f"Saved generated image to {output_path}")

        else:
            print(f"Invalid mode: {mode}. Please choose 'img2img' or 'text2img'.")

# Example usage: process rows 0 to 5
process_prompts(0, 2, mode="text2img")  # Change to `mode="text2img"` to test text-to-image generation

# load img
# img = Image.open("../data/EVA.jpg")

# gen = ImageGenerator()
# new_img = gen.generate(img, prompt="The huge robot is running",)

# new_img.save("output.jpg")