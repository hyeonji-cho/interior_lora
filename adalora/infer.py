from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, MultiControlNetModel
import torch
from PIL import Image
from transformers import pipeline, BeitImageProcessor, BeitForSemanticSegmentation, OneFormerProcessor, OneFormerForUniversalSegmentation
import numpy as np
import warnings


init_image = Image.open("./train_dataset/modern/modern_40.jpg").convert("RGB")

# 1. Depth map 생성
depth_pipe = pipeline("depth-estimation")
depth_map = depth_pipe(init_image)["depth"]
depth_map.save("./depth_imgs/test_depth.png")

# 2-1 oneformer segmentation map 생성
processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
model = model.to("cuda")

# 2-1 oneformer segmentation map
# task_inputs 추가 - "semantic" 작업 지정
inputs = processor(
    images=init_image, 
    task_inputs=["semantic"],  # 이 부분이 필수!
    return_tensors="pt"
)
inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

# OneFormer는 post_process_semantic_segmentation 사용
predicted_semantic_map = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[init_image.size[::-1]]
)[0]

label_map = predicted_semantic_map.cpu().numpy()

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

# Color segmentation map 생성
seg_color = ADE20K_PALETTE[label_map]
segmentation_map = Image.fromarray(seg_color).resize(init_image.size, Image.NEAREST)
segmentation_map.save("./seg_imgs/test_segmentation_beit.png")

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

# 스케줄러를 DPMSolverMultistepScheduler로 변경 (더 안정적)
from diffusers import DPMSolverMultistepScheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# VAE 디코딩 최적화
pipe.enable_vae_tiling()
pipe.enable_vae_slicing()

# AdaLoRA 가중치 로딩 - 전용 로더 사용
from adalora_loader import AdaLoRALoader

print("=== Loading AdaLoRA weights ===")
adalora_loader = AdaLoRALoader()
checkpoint_path = "./final-lora-weight/adalora_south_30/checkpoint-4000"

# AdaLoRA 가중치를 파이프라인에 로드 시도
if adalora_loader.load_to_pipeline(pipe, checkpoint_path):
    print("✓ AdaLoRA weights successfully applied to pipeline")
else:
    print("⚠ Using base model without AdaLoRA weights")


# 이미지 크기 조정
init_image = init_image.resize((1024, 1024))
depth_map = depth_map.resize((1024, 1024))
segmentation_map = segmentation_map.resize((1024, 1024))

# 프롬프트 - 학습된 스타일을 더 강하게 적용
V = "boisversionsrai"
prompt = f"a {V} style interior room, photorealistic, highly detailed"
negative_prompt = "cartoon, anime, illustration, painting, sketch, surreal, unrealistic, low quality, blurry"

# 생성 - 검은 이미지 문제 해결을 위한 설정 조정
print("Starting image generation...")
print(f"Init image size: {init_image.size}")
print(f"Depth map size: {depth_map.size}")
print(f"Segmentation map size: {segmentation_map.size}")

with torch.no_grad():
    generated_images = pipe(
        seed=42,
        prompt=prompt,
        image=init_image,
        control_image=[depth_map, segmentation_map],
        controlnet_conditioning_scale=[0.7, 0.8], # ControlNet 강도 더 낮춤 (AdaLoRA 영향력 증대)
        strength=1.0,  # img2img 강도 약간 낮춤
        guidance_scale=15.0,  # CFG 더 낮춤 (AdaLoRA 스타일 강화)
        num_inference_steps=100,  # 스텝 수 줄임
        generator=torch.Generator("cuda").manual_seed(123),  # 다른 시드 시도
        eta=1.0,  # 노이즈 스케줄링 파라미터
    ).images

# 이미지 후처리 - NaN이나 극값 제거
def clean_image(image):
    """이미지에서 NaN이나 극값을 제거"""
    if hasattr(image, 'getdata'):  # PIL Image
        # PIL Image를 numpy로 변환하여 정리
        img_array = np.array(image)
        # NaN을 0으로 대체
        img_array = np.nan_to_num(img_array, nan=0.0, posinf=255.0, neginf=0.0)
        # 값 범위를 [0, 255]로 클립
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        # 다시 PIL Image로 변환
        return Image.fromarray(img_array)
    
    # numpy array인 경우
    if isinstance(image, np.ndarray):
        # NaN을 0으로 대체
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        # 값 범위를 [0, 1]로 클립
        image = np.clip(image, 0.0, 1.0)
        return image
    
    return image

# 생성된 이미지 정리 및 저장
print(f"Generated {len(generated_images)} images")
for i, img in enumerate(generated_images):
    # 이미지 정리 적용
    cleaned_img = clean_image(img)
    
    # 이미지 품질 검증
    img_array = np.array(cleaned_img)
    avg_brightness = np.mean(img_array)
    print(f"Image {i}: Average brightness = {avg_brightness:.1f}")
    
    if avg_brightness < 10:
        print(f"Warning: Image {i} appears to be very dark (brightness: {avg_brightness:.1f})")
    elif avg_brightness > 245:
        print(f"Warning: Image {i} appears to be very bright (brightness: {avg_brightness:.1f})")
    else:
        print(f"Image {i} appears normal")
    
    # 정리된 이미지 저장
    filename = f"./generated_imgs/adalora_south_4000.jpg"
    cleaned_img.save(filename)
    print(f"Saved cleaned image: {filename}") 