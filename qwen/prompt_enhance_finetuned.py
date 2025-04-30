import json
import random
from PIL import Image, ImageDraw, ImageFont
import base64
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# System prompt
sys_prompt = '''You are a prompt enhancer for image generation models like DALL·E or Midjourney, fine-tuned specifically for prompt improvement tasks.

Given:

An image.

The original text prompt used to generate the image.

Your task:

Carefully analyze both the image and the original prompt.

Generate a more detailed, vivid, and compositionally rich enhanced prompt.

Your enhanced prompt should:

Clearly describe specific objects, artistic styles, settings, lighting, mood, and artistic techniques present in the image.

Be clear, descriptive, and optimized for AI image generation.

Expand upon vague or minimal descriptions from the original prompt.

Maintain a total length equal to or fewer than 77 tokens.

Always ensure your enhanced prompt accurately reflects details observed in both the provided image and original prompt.

Input:

Image: file://path/to/image

Original Prompt: 'your original prompt'

Output:

Enhanced Prompt: '[Your improved prompt here]'
'''

# Inference function
def inference(image_path, prompt, sys_prompt=sys_prompt, max_new_tokens=77, return_input=False):
    image = Image.open(image_path)
    image_local_path = "file://" + image_path
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"image": image_local_path},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if return_input:
        return output_text[0], inputs
    else:
        return output_text[0]

# Load model and processor
checkpoint = "../output/checkpoint"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    checkpoint, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", device_map="auto"
)
processor = AutoProcessor.from_pretrained(checkpoint)

# ====== CONFIGURATION ======
INPUT_CSV_PATH = "../datasets/900k-diffusion-prompts-dataset/finetune/test/test.csv"
OUTPUT_CSV_PATH = "../output/enhanced_prompts_finetuned3.csv"
NUM_SAMPLES = 750
IMAGE_SAVE_DIR = "../datasets/900k-diffusion-prompts-dataset/downloaded_images"

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# Load and subset input CSV
df = pd.read_csv(INPUT_CSV_PATH).head(NUM_SAMPLES)

# Store results
records = []

for idx, row in df.iterrows():
    image_url = row['url']
    image_path = os.path.join(IMAGE_SAVE_DIR, f"{idx}.png")
    raw_prompt = row['prompt']
    original_prompt = ' '.join(str(raw_prompt).split())  # flatten multi-line

    try:
        # Download image if needed
        if not os.path.exists(image_path):
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            image.save(image_path)
        else:
            print(f"Image {idx}.png already exists. Skipping download.")

        # Run inference
        model_response = inference(image_path, original_prompt)
        print(f"[{idx}] model_response: {model_response}")

        # Extract prompt
        if isinstance(model_response, str) and "Enhanced Prompt:" in model_response:
            enhanced_prompt = model_response.split("Enhanced Prompt:")[-1].strip().strip('"')
        else:
            print(f"Warning: No enhanced prompt for row {idx}")
            continue

        if len(enhanced_prompt.split()) < 25:
            print(f"Warning: Too short ({len(enhanced_prompt.split())} tokens) for row {idx}")
            continue

        # Append to results
        records.append({
            "id": idx,
            "url": row.get("url", ""),
            "width": row.get("width", ""),
            "height": row.get("height", ""),
            "source_site": row.get("source_site", ""),
            "similarity": row.get("similarity", ""),
            "original_prompt": original_prompt,
            "enhanced_prompt": enhanced_prompt
        })

    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        continue

# Save final CSV
final_df = pd.DataFrame(records)
os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
final_df.to_csv(OUTPUT_CSV_PATH, index=False)

# Summary
print(f"Intended to process {NUM_SAMPLES} rows.")
print(f"Successfully processed and saved {len(final_df)} rows with at least 25 tokens.")
print(f"CSV saved to: {OUTPUT_CSV_PATH}")
