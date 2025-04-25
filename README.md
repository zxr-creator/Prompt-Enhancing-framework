# Prompt-Enhancing-framework
Cource project of Comp 646

Qwen finetuning:

1. Clip filter
   ```
   python clip_filter.py \
    --input_csv ./datasets/900k-diffusion-prompts-dataset/diffusion_prompts.csv \
    --output_dir ./datasets/900k-diffusion-prompts-dataset/finetune \
    --batch_size 64 \
    --num_workers 32
   ```
2. Finetune qwen with filtered data
3. Evaluation