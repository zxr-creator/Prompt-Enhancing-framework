# Prompt-Enhancing-framework
Cource project of Comp 646

Qwen finetuning:
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
