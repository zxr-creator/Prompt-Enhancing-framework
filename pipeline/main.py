import pandas as pd
from ImageGenerator import ImageGenerator, TextToImageGenerator  # Ensure this is imported from your ImageGenerator module
from PIL import Image
import requests
from io import BytesIO
from huggingface_hub import login  # Import Hugging Face login
import yaml
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import ChunkedEncodingError, Timeout, ConnectionError, HTTPError

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

def create_session_with_retries(retries=3, backoff=0.5, status_codes=[500, 502, 503, 504]):
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=status_codes,
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def download_image(url, timeout=(5, 60)):
    """
    Download image from a URL with retry and timeout.
    Args:
        url (str): Image URL
        timeout (tuple): (connect_timeout, read_timeout)
    Returns:
        PIL.Image.Image or None
    """
    session = create_session_with_retries()
    try:
        with session.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()  # Raise if 4xx or 5xx
            content = response.raw.read(decode_content=True)
            img = Image.open(BytesIO(content))
            return img
    except (ChunkedEncodingError, Timeout, ConnectionError, HTTPError) as e:
        print(f"[Download Failed] {url} — {e}")
        return None


def process_prompts(start_row, end_row, mode="img2img"):
    """
    Process prompts and generate images.

    Args:
        start_row (int): Start row index in the CSV file.
        end_row (int): End row index in the CSV file.
        mode (str): Mode of generation, either "img2img" or "text2img".
    """
    os.makedirs("output", exist_ok=True) # Ensure output directory exists
    # Load prompts and image URLs from CSV
    data = pd.read_csv("../output/enhanced_prompts.csv")
    print(data.shape)
    selected_data = data.iloc[start_row:end_row]  # Select specific rows
    
    # Preload images for img2img mode
    images = {}
    if mode == "img2img":
        for index, row in selected_data.iterrows():
            image_url = row['url']
            img = download_image(image_url)
            if img is not None:
                images[index] = img
                print(f"Preloaded image for index {index}")
            else:
                print(f"Skipping index {index} due to download failure.")
                images[index] = None
            #response = requests.get(image_url)
            # if response.status_code == 200:
            #     img = Image.open(BytesIO(response.content))
            #     images[index] = img
            #     print(f"Preloaded image for index {index}")
            # else:
            #     print(f"Failed to download image from {image_url}")
            #     images[index] = None
        print("======= Load finished =======")
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
            output_path = f"output/output_img2img_{index}.jpg"
            new_img.save(output_path)
            print(f"Saved generated image to {output_path}")

        elif mode == "text2img":
            # Generate image directly from text prompt
            gen = TextToImageGenerator()
            new_img = gen.generate(prompt=prompt)

            # Save the generated image
            output_path = f"output/output_text2img_{index}.jpg"
            new_img.save(output_path)
            print(f"Saved generated image to {output_path}")

        else:
            print(f"Invalid mode: {mode}. Please choose 'img2img' or 'text2img'.")

# Example usage: process rows 0 to 5
authenticate_huggingface_from_yaml("../API_keys.yaml")  # Authenticate with Hugging Face
process_prompts(0, 100, mode="img2img")  # Change to `mode="text2img"` to test text-to-image generation
