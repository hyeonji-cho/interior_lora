import os
from glob import glob
from PIL import Image
import torch
import numpy as np
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, MultiControlNetModel
from transformers import pipeline, OneFormerProcessor, OneFormerForUniversalSegmentation

# 경로 설정
input_dir = "./test_imgs"
depth_out_dir = "./depth_imgs"
seg_out_dir = "./seg_imgs"
final_out_dir = "./generated_imgs"
os.makedirs(depth_out_dir, exist_ok=True)
os.makedirs(seg_out_dir, exist_ok=True)
os.makedirs(final_out_dir, exist_ok=True)

# Depth pipeline 초기화
depth_pipe = pipeline("depth-estimation")

# Segmentation 모델 초기화
processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large").to("cuda")

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

# ControlNet 모델 초기화
controlnets = [
    ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16),
    ControlNetModel.from_pretrained("SargeZT/sdxl-controlnet-seg", torch_dtype=torch.float16)
]
multi_controlnet = MultiControlNetModel(controlnets)

pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=multi_controlnet,
    torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights(".", weight_name="./final-lora-weight/coastal/checkpoint-3000/") # weight 이름 넣기

# ADE20K 팔레트 정의 생략 (기존 코드 그대로 사용)

# 이미지 목록 가져오기
image_paths = glob(os.path.join(input_dir, "*.jpg"))

for img_path in image_paths:
    filename = os.path.splitext(os.path.basename(img_path))[0]

    init_image = Image.open(img_path).convert("RGB")

    # 1. Depth 생성
    depth_map = depth_pipe(init_image)["depth"]
    depth_map_path = os.path.join(depth_out_dir, f"{filename}_depth.png")
    depth_map.save(depth_map_path)

    # 2. Segmentation 생성
    inputs = processor(images=init_image, task_inputs=["semantic"], return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_semantic_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[init_image.size[::-1]]
    )[0]
    label_map = predicted_semantic_map.cpu().numpy()
    seg_color = ADE20K_PALETTE[label_map]
    segmentation_map = Image.fromarray(seg_color).resize(init_image.size, Image.NEAREST)
    seg_path = os.path.join(seg_out_dir, f"{filename}_seg.png")
    segmentation_map.save(seg_path)

    # 리사이즈
    resized_init = init_image.resize((1024, 1024))
    resized_depth = depth_map.resize((1024, 1024))
    resized_seg = segmentation_map.resize((1024, 1024))

    # 프롬프트 설정
    V = "boisversionsrai"

    # depth, seg conditioning scale 모든 조합에 대해 이미지 생성
    depth_scales = [0.5]
    seg_scales = [0.6]
    generator=torch.Generator("cuda").manual_seed(123)
    prompt = f"a {V} style interior room, photorealistic, highly detailed"
    negative_prompt = "cartoon, anime, illustration, painting, sketch, surreal, unrealistic, low quality, blurry"
    
    for p_idx, prompt in enumerate(prompts):
        for d_scale in depth_scales:
            for s_scale in seg_scales:
                generated_images = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=resized_init,
                    control_image=[resized_depth, resized_seg],
                    controlnet_conditioning_scale=[d_scale, s_scale], #depth, seg
                    strength=1.0,
                    guidance_scale=15.0,
                    num_inference_steps=100,
                    generator=generator,
                    eta=1.0
                ).images
                # 파일명에 프롬프트 인덱스, depth, seg scale 포함
                out_path = os.path.join(final_out_dir, f"generated_{p_idx}_{d_scale}_{s_scale}.jpg")
                generated_images[0].save(out_path)
