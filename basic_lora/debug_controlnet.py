#!/usr/bin/env python3
"""
Control 이미지 및 파이프라인 디버깅 스크립트
"""

import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
from transformers import pipeline

print("=== Control Image Debug Test ===")

# 1. 원본 이미지 로드 및 분석
print("Loading and analyzing input image...")
init_image = Image.open("./train_dataset/modern/modern_40.jpg").convert("RGB")
print(f"Original image: {init_image.mode}, {init_image.size}")

# 2. Depth map 생성 및 분석
print("Generating depth map...")
depth_pipe = pipeline("depth-estimation")
depth_map = depth_pipe(init_image)["depth"]

# Depth map 분석
depth_array = np.array(depth_map)
print(f"Depth map shape: {depth_array.shape}")
print(f"Depth values - Min: {np.min(depth_array)}, Max: {np.max(depth_array)}, Mean: {np.mean(depth_array)}")

# Depth map 저장
depth_map.save("./debug_depth.jpg")
print("Depth map saved as 'debug_depth.jpg'")

# 3. 단일 ControlNet으로 테스트
print("Loading single ControlNet pipeline...")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0", 
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# 4. 이미지 리사이즈 및 전처리
target_size = (768, 768)  # 더 작은 크기로 테스트
init_image_resized = init_image.resize(target_size)
depth_map_resized = depth_map.resize(target_size)

# Depth map을 grayscale로 변환
if depth_map_resized.mode != 'L':
    depth_map_resized = depth_map_resized.convert('L')

print(f"Resized images: {init_image_resized.size}, depth: {depth_map_resized.size}")

# 5. 매우 단순한 설정으로 생성
print("Generating with single ControlNet...")
prompt = "modern interior design, living room"
negative_prompt = "dark, black"

try:
    result_images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image_resized,
        control_image=depth_map_resized,
        controlnet_conditioning_scale=0.5,
        strength=0.7,
        guidance_scale=7.5,
        num_inference_steps=20,
        generator=torch.manual_seed(42)
    ).images
    
    # 결과 분석
    for i, img in enumerate(result_images):
        img_array = np.array(img)
        brightness = np.mean(img_array)
        print(f"Result image {i}: brightness = {brightness}")
        
        if brightness < 10:
            print(f"WARNING: Image {i} is mostly black!")
        else:
            print(f"Image {i} looks normal")
        
        img.save(f"./debug_single_controlnet_{i}.jpg")
        print(f"Saved debug_single_controlnet_{i}.jpg")

except Exception as e:
    print(f"Error during generation: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Debug test completed ===")

# 6. 추가 정보 출력
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
