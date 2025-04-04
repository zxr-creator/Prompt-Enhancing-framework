"""Get evaluation results for images using the evaluation module.
"""



import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from huggingface_hub import login  # Import Hugging Face login
import yaml
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import ChunkedEncodingError, Timeout, ConnectionError, HTTPError
from evaluation import ImageQualityEvaluator


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

def evaluate_results(start_row, end_row):
    """
    Process prompts and evaluate images.

    Args:
        start_row (int): Start row index in the CSV file.
        end_row (int): End row index in the CSV file.
        mode (str): Mode of generation, either "img2img" or "text2img".
    """
    
    # Load prompts and image URLs from CSV
    data = pd.read_csv("./enhanced_prompts.csv")
    print(data.shape)
    selected_data = data.iloc[start_row:end_row]  # Select specific rows
    
    images_original = {}
    images_generated = {}
    
    for index, row in selected_data.iterrows():
        image_url = row['url']
        img = download_image(image_url)
        if img is not None:
            images_original[index] = img
            print(f"Preloaded image for index {index}")
        else:
            print(f"Skipping index {index} due to download failure.")
            images_original[index] = None
    # Load generated image
        gen_img_path = f"./pipeline/output/output_img2img_{index}.jpg"
        if os.path.exists(gen_img_path):
            images_generated[index] = Image.open(gen_img_path)
            print(f"Loaded generated image for index {index}")
        else:
            print(f"Generated image not found for index {index}")
            images_generated[index] = None
    print("======= Images loading complete =======")

    # Clear terminal
    # os.system('cls' if os.name == 'nt' else 'clear')
    # print("======= Load finished, Start Evaluation =======")
    # Initialize the image quality evaluator
    evaluator = ImageQualityEvaluator()

    # Prepare image lists, filtering out any failed (None) downloads
    original_list = [img for img in images_original.values() if img is not None]
    generated_list = [img for img in images_generated.values() if img is not None]

    print(f"Original valid images: {len(original_list)}, Generated valid images: {len(generated_list)}")

    # Compute Inception Score for original images
    if len(original_list) > 0:
        original_probs = evaluator.get_inception_probs(original_list)
        original_is = evaluator.compute_is(original_probs)
        print(f"Inception Score (Original): {original_is:.4f}")
    else:
        print("No valid original images to evaluate IS.")

    # Compute Inception Score for generated images
    if len(generated_list) > 0:
        generated_probs = evaluator.get_inception_probs(generated_list)
        generated_is = evaluator.compute_is(generated_probs)
        print(f"Inception Score (Generated): {generated_is:.4f}")
    else:
        print("No valid generated images to evaluate IS.")

    # Prepare data for CLIP Score computation (only for generated images)
    text_prompts = [selected_data.loc[i, "prompt"] for i in images_generated.keys() if images_generated[i] is not None]
    valid_generated_images = [images_generated[i] for i in images_generated.keys() if images_generated[i] is not None]

    # Compute CLIP Score between generated images and their text prompts
    if len(valid_generated_images) > 0:
        clip_score = evaluator.compute_clip_score(valid_generated_images, text_prompts)
        print(f"CLIP Score (Generated vs Prompt): {clip_score:.4f}")
    else:
        print("No valid generated images for CLIP score.")

if __name__ == "__main__":
    """
    Main entry point for the script.
    Adjust start_row, end_row, and mode as needed.
    """
    # Example usage
    start_row = 0
    end_row = 100  # Process first 10 rows for testing

    evaluate_results(start_row, end_row)

