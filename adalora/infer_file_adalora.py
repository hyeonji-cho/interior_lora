from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, MultiControlNetModel
import torch
from PIL import Image
from transformers import pipeline, BeitImageProcessor, BeitForSemanticSegmentation, OneFormerProcessor, OneFormerForUniversalSegmentation
import numpy as np
import warnings
import os
import glob


# generated_imgs/coastal_30 디렉토리의 모든 이미지 파일 찾기
image_dir = "./train_dataset/modern"
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
image_files = []

for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))

if not image_files:
    print(f"No images found in {image_dir}")
    exit()

print(f"Found {len(image_files)} images in {image_dir}")

# Depth estimation pipeline 초기화
depth_pipe = pipeline("depth-estimation")

# OneFormer segmentation 모델 초기화
processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
model = model.to("cuda")

# ADE20K 색상 팔레트
ADE20K_PALETTE = np.array([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
    [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
    [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
    [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
    [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
    [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
    [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
    [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
    [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
    [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
    [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
    [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
    [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
    [102, 255, 0], [92, 0, 255]
], dtype=np.uint8)

# Multi-ControlNet 설정
controlnets = [
    ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16),
    ControlNetModel.from_pretrained("SargeZT/sdxl-controlnet-seg", torch_dtype=torch.float16)
]
multi_controlnet = MultiControlNetModel(controlnets)

# SDXL + Multi-ControlNet 파이프라인
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=multi_controlnet,
    torch_dtype=torch.float16
).to("cuda")

# 스케줄러를 DPMSolverMultistepScheduler로 변경
from diffusers import DPMSolverMultistepScheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# VAE 최적화
pipe.enable_vae_tiling()
pipe.enable_vae_slicing()

# AdaLoRA 가중치 로딩
from adalora_loader import AdaLoRALoader

print("=== Loading AdaLoRA weights ===")
adalora_loader = AdaLoRALoader()
checkpoint_path = "./final-lora-weight/adalora_coastal_30/checkpoint-2500"

if adalora_loader.load_to_pipeline(pipe, checkpoint_path):
    print("✓ AdaLoRA weights successfully applied to pipeline")
else:
    print("⚠ Using base model without AdaLoRA weights")

# 이미지 후처리 함수
def clean_image(image):
    """이미지에서 NaN이나 극값을 제거"""
    if hasattr(image, 'getdata'):  # PIL Image
        img_array = np.array(image)
        img_array = np.nan_to_num(img_array, nan=0.0, posinf=255.0, neginf=0.0)
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    if isinstance(image, np.ndarray):
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = np.clip(image, 0.0, 1.0)
        return image
    
    return image

def process_segmentation(init_image):
    """OneFormer로 segmentation map 생성"""
    inputs = processor(
        images=init_image, 
        task_inputs=["semantic"],
        return_tensors="pt"
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_semantic_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[init_image.size[::-1]]
    )[0]

    label_map = predicted_semantic_map.cpu().numpy()
    seg_color = ADE20K_PALETTE[label_map]
    segmentation_map = Image.fromarray(seg_color).resize(init_image.size, Image.NEAREST)
    
    return segmentation_map

# 프롬프트 설정
V = "boisversionsrai"
prompt = f"a {V} style interior room, photorealistic, highly detailed"
negative_prompt = "cartoon, anime, illustration, painting, sketch, surreal, unrealistic, low quality, blurry"

# 각 이미지에 대해 처리
for idx, image_path in enumerate(image_files):
    print(f"\n=== Processing image {idx+1}/{len(image_files)}: {os.path.basename(image_path)} ===")
    
    try:
        # 이미지 로드
        init_image = Image.open(image_path).convert("RGB")
        init_image = init_image.resize((1024, 1024))
        
        print("Generating depth map...")
        # Depth map 생성
        depth_map = depth_pipe(init_image)["depth"]
        depth_map = depth_map.resize((1024, 1024))
        
        # Depth map 저장 (선택사항)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        depth_map.save(f"./depth_imgs/{base_name}_depth.png")
        
        print("Generating segmentation map...")
        # Segmentation map 생성
        segmentation_map = process_segmentation(init_image)
        segmentation_map = segmentation_map.resize((1024, 1024))
        
        # Segmentation map 저장 (선택사항)
        segmentation_map.save(f"./seg_imgs/{base_name}_seg.png")
        
        print("Generating final image...")
        # 이미지 생성
        with torch.no_grad():
            generated_images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                control_image=[depth_map, segmentation_map],
                controlnet_conditioning_scale=[0.7, 0.7],
                strength=1.0,
                guidance_scale=12.0,
                num_inference_steps=100,
                generator=torch.Generator("cuda").manual_seed(42),  # 각 이미지마다 다른 시드
            ).images
        
        # 생성된 이미지 저장
        for i, img in enumerate(generated_images):
            cleaned_img = clean_image(img)
            
            # 이미지 품질 검증
            img_array = np.array(cleaned_img)
            avg_brightness = np.mean(img_array)
            print(f"Generated image brightness: {avg_brightness:.1f}")
            
            if avg_brightness < 10:
                print(f"Warning: Generated image appears very dark")
            elif avg_brightness > 245:
                print(f"Warning: Generated image appears very bright")
            else:
                print(f"Generated image appears normal")
            
            # 결과 이미지 저장
            output_filename = f"./generated_imgs/coastal_30/coastal_2500_{base_name}_{i}.jpg"
            cleaned_img.save(output_filename)
            print(f"✓ Saved result: {output_filename}")
            
    except Exception as e:
        print(f"❌ Error processing {image_path}: {str(e)}")
        continue

print(f"\n🎉 Completed processing all {len(image_files)} images!")
print("Results saved")