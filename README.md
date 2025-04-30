# Prompt-Enhancing-framework
Cource project of Comp 646

Recent advances in generative AI have enabled models like Stable Diffusion to support both textto- image and image-to-image generation, producing high-quality images from textual prompts and visual references. However, generating effective prompts remains a challenge, often resulting in suboptimal outputs. In this progress report, we present a Prompt Enhancing framework that leverages a finetuned MultiModal Large Language Model (MLLM), Qwen2.5-VL, to refine user-provided prompts based on both textual and visual inputs. The enhanced prompts are then used in a Stable Diffusion generation pipeline. Experiments on a subset of the 900k Diffusion Prompts Dataset demonstrate that our enhanced prompts significantly improve image quality and semantic alignment. Project code is available on GitHub.

The whole pipeline:
1. Clip filter
```
python clip_filter.py \
    --input_csv ./datasets/900k-diffusion-prompts-dataset/diffusion_prompts.csv \
    --output_dir ./datasets/900k-diffusion-prompts-dataset/finetune \
    --train_ratio 0.8 \
    --batch_size 128 \
    --num_workers 64
```
2. Qwen finetune
```
python converter.py
bash finetune.sh 
cd qwen
python prompt_enhance_finetuned.py
```
3. Evaluation 
```
cd ./pipeline
python main.py
python eval_result.py
```