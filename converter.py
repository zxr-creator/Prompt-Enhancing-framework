import csv, json, os
import pandas as pd
import argparse
from transformers import AutoTokenizer

SYSTEM_MESSAGE = """You are a prompt enhancer for image generation models like DALL·E or Midjourney. Given: An image.

The original text prompt used to generate the image.

Your task is to analyze the image and the original prompt, and then output a more detailed, vivid, and compositionally rich enhanced prompt that would recreate the image with higher fidelity, aesthetic quality, and visual richness. The enhanced prompt should:

Include specific objects, styles, settings, lighting, mood, and artistic techniques observed in the image.

Be clear, descriptive, and suitable for input into an AI image generator.

Improve upon vague or minimal descriptions in the original prompt.

The Enhanced Prompt should be equal to or less than 77 tokens. You should be aware of that.

Input:

Image: file://path/to/image

Original Prompt: 'your original prompt'

Output:

Enhanced Prompt: '[Your improved prompt here]'
"""

train_csv_path   = "./datasets/900k-diffusion-prompts-dataset/finetune/train/train.csv"
train_jsonl_path = "./datasets/900k-diffusion-prompts-dataset/finetune/train/qwen_train.jsonl"

test_csv_path   = "./datasets/900k-diffusion-prompts-dataset/finetune/test/test.csv"
test_jsonl_path = "./datasets/900k-diffusion-prompts-dataset/finetune/test/qwen_test.jsonl"

MAX_TOK = 77
MIN_TOK = 55

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)

def truncate_prompt(prompt: str, max_tokens: int = MAX_TOK) -> str:
    """Safely truncate a prompt to max_tokens using the model tokenizer."""
    encoded = tokenizer(prompt, truncation=True, max_length=max_tokens, return_tensors=None)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
    return decoded.strip()

def prompt_token_count(prompt: str) -> int:
    """Return the number of tokens in a prompt."""
    encoded = tokenizer(prompt, return_tensors=None)
    return len(encoded["input_ids"])

def convert(csv_path, out_path):
    df = pd.read_csv(csv_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in df.itertuples():
            # Count original prompt tokens
            token_count = prompt_token_count(row.prompt)
            if token_count < MIN_TOK:
                continue  # Skip prompts that are too short

            truncated_prompt = truncate_prompt(row.prompt)

            conv = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                {"role": "user", "content": [
                    {"type": "image", "image": row.url},
                    {"type": "text", "text": f"Original Prompt: '{row.prompt}'"}
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": f"Enhanced Prompt: '{truncated_prompt}'"}]}
            ]
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

# Convert both train and test sets
convert(train_csv_path, train_jsonl_path)
convert(test_csv_path, test_jsonl_path)
