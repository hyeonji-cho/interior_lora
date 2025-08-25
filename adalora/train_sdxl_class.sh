#!/usr/bin/env bash
set -e

EXAMPLES_DIR="/data3/hyeonji/workspace/pretrained_lora"  # diffusers 실제 경로
V="boisversionsrai"

accelerate launch "$EXAMPLES_DIR/train_dreambooth_lora_sdxl.py" \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
  --instance_data_dir=./gpt_tropical \
  --instance_prompt="a ${V} interior room with ${V} furniture" \
  --class_data_dir=./general_upsampled \
  --class_prompt="a interior room with furniture" \
  --with_prior_preservation \
  --resolution=1024 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --lr_scheduler=constant \
  --max_train_steps=2500 \
  --checkpointing_steps=100 \
  --checkpoints_total_limit 2 \
  --output_dir=./final-lora-weight/tropical_out_gpt \
  --report_to=tensorboard \
  --validation_prompt="a ${V} interior room with ${V} furniture" \
  --validation_epochs=1 \
  --seed=42 \
  --use_8bit_adam \
  --mixed_precision=fp16 \
  --prior_loss_weight=0.3 \
  --num_class_images=150 \
  --center_crop \
  --train_text_encoder \
  --rank=32 \
  --logging_dir=./logs/tropical_run_1 \
  --enable_xformers_memory_efficient_attention