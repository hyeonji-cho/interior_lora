#!/usr/bin/env bash
set -e

EXAMPLES_DIR="/data3/hyeonji/workspace/pretrained_lora"
V="boisversionsrai"

# accelerate launch "$EXAMPLES_DIR/train_dreambooth_lora_sdxl.py" \
#   --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
#   --instance_data_dir=./southwestern900 \
#   --instance_prompt="a ${V} interior room" \
#   --resolution=1024 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --learning_rate=1e-4 \
#   --lr_scheduler=constant \
#   --max_train_steps=2500 \
#   --checkpointing_steps=100 \
#   --checkpoints_total_limit=2 \
#   --output_dir=./final-lora-weight/southwestern_output_5 \
#   --report_to=tensorboard \
#   --validation_prompt="a ${V} interior room" \
#   --validation_epochs=1 \
#   --seed=42 \
#   --use_8bit_adam \
#   --mixed_precision=fp16 \
#   --center_crop \
#   --train_text_encoder \
#   --rank=10 \
#   --enable_xformers_memory_efficient_attention
# --resume_from_checkpoint "./final-lora-weight/southwestern_lora/checkpoint-2500/pytorch_lora_weights.safetensors" \

accelerate launch "$EXAMPLES_DIR/train_dreambooth_lora_sdxl.py" \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
  --instance_data_dir "./train_dataset/gpt_tropical" \
  --output_dir "./final-lora-weight/adalora_gpt_tropical_rank_16" \
  --instance_prompt "a ${V} style interior" \
  --resolution 768 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --max_train_steps 10000 \
  --rank 32 \
  --mixed_precision fp16 \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps 1000 \
  --checkpoints_total_limit 10 \
  --learning_rate 1e-4 \
  --target_rank 16 \
  --resume_from_checkpoint "./final-lora-weight/adalora_gpt_tropical_rank_16/checkpoint-5000" \
  --use_adalora
