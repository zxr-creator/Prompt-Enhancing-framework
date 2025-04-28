from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from PIL import Image; import requests, torch, json, qwen_vl_utils

sys_prompt = """You are a prompt enhancer for image generation models like DALL·E or Midjourney.Given:An image.

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
"""

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "qwen2.5vl-promptenh-full", torch_dtype=torch.bfloat16).cuda()
proc  = Qwen2_5_VLProcessor.from_pretrained("qwen2.5vl-promptenh-full")

url = "https://image.lexica.art/full_jpg/8c7b3ffa-0dc1-4335-92d9-809eeed49093"
img = Image.open(requests.get(url, stream=True).raw)

messages = [
  {"role":"system","content":[{"type":"text","text":sys_prompt}]},
  {"role":"user","content":[
     {"type":"image","image":img},
     {"type":"text","text":"Original Prompt: 'Happy customer feeling relief and in love with a pineapple during shopping'"}]}]

text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, vis_infos = qwen_vl_utils.process_vision_info(messages)
batch = proc(text=[text], images=images, vision_infos=vis_infos,
             return_tensors="pt").to(model.device)

out = model.generate(**batch, max_new_tokens=128)
print(proc.batch_decode(out[:, batch.input_ids.shape[1]:], skip_special_tokens=True)[0])
