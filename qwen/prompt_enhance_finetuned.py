import json
import random
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
import base64
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

sys_prompt = '''You are a prompt enhancer for image generation models like DALL·E or Midjourney.Given:An image.

The original text prompt used to generate the image.

Your task is to analyze the image and the original prompt, and then output a more detailed, vivid, and compositionally rich enhanced prompt that would recreate the image with higher fidelity, aesthetic quality, and visual richness. The enhanced prompt should:

Include specific objects, styles, settings, lighting, mood, and artistic techniques observed in the image.

Be clear, descriptive, and suitable for input into an AI image generator.

Improve upon vague or minimal descriptions in the original prompt.

The Enhanced Prompt should be euqal to or less than 77 tokens. You should be aware of that 

Input:

Image: [insert image]

Original Prompt: '[insert original prompt here]'

Output:

Enhanced Prompt: '[Your improved prompt here]'
'''

# @title inference function
def inference(image_path, prompt, sys_prompt=sys_prompt, max_new_tokens=4096, return_input=False):
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
    # print("text:", text)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if return_input:
        return output_text[0], inputs
    else:
        return output_text[0]
    
#  base 64 
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

checkpoint = "../output/checkpoint/checkpoint-479"

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(checkpoint)

# ====== CONFIGURATION ======
INPUT_CSV_PATH = "../datasets/900k-diffusion-prompts-dataset/finetune/test/test.csv"
OUTPUT_CSV_PATH = "../output/enhanced_prompts_finetuned.csv"
NUM_SAMPLES = 100  # <--- Change this number to control how many rows are processed
IMAGE_SAVE_DIR = "../datasets/900k-diffusion-prompts-dataset/downloaded_images"

# ====== Ensure image directory exists ======
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# ====== Load the CSV ======
df = pd.read_csv(INPUT_CSV_PATH)

# ====== Slice for the desired number of rows ======
df_subset = df.head(NUM_SAMPLES)

# ====== Store successful rows and enhanced prompts ======
successful_rows = []
enhanced_prompts = []

# ====== Process each row ======
for idx, row in df_subset.iterrows():
    prompt = row['prompt']
    image_url = row['url']
    image_id = row['id']
    image_path = os.path.join(IMAGE_SAVE_DIR, f"{image_id}.png")
    
    try:
        # Check if image already exists, else download it
        if not os.path.exists(image_path):
            response = requests.get(image_url)
            response.raise_for_status()  # ensure it's a valid response
            image = Image.open(BytesIO(response.content))
            image.save(image_path)
        else:
            print(f"Image {image_id}.png already exists. Skipping download.")

        # Run inference
        model_response = inference(image_path, prompt)
        print(f"model_response: {model_response}")

        # Save successful row and enhanced prompt
        enhanced_prompts.append(model_response)
        successful_rows.append(row)

    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        continue  # skip this row

# ====== Build final DataFrame and save ======
final_df = pd.DataFrame(successful_rows)
final_df['prompt'] = enhanced_prompts  # replace with enhanced prompts

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

final_df.to_csv(OUTPUT_CSV_PATH, index=False)

# ====== Summary ======
print(f"Intended to process {NUM_SAMPLES} rows.")
print(f"Successfully processed and saved {len(final_df)} rows.")
print(f"Enhanced CSV saved to {OUTPUT_CSV_PATH}")
