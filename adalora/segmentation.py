import torch
from transformers import pipeline
from PIL import Image
import numpy as np

device = torch.device("cpu")

# 1. 이미지 로드
image = Image.open("modern_820.jpg").convert("RGB")

# 2. 세그멘테이션 수행
seg_pipe = pipeline("image-segmentation", model="nvidia/segformer-b4-finetuned-ade-512-512", device=-1)
segments = seg_pipe(image)

# 3. 선택할 클래스 지정
selected_labels = {
    "wall", "floor", "ceiling", "door", "sofa", "cabinet","chair","table", "lamp", "curtain",
    "rug", "bookshelf","tv","painting", # 거실
    "pillow", "curtain", "bed", # 침실
    "sink", "stove", "refrigerator", # 주방
    "bathtub","sink","mirror", # 욕실
    
}

# 4. 마스크 합치기
combined_mask = None
for seg in segments:
    if seg["label"] in selected_labels:
        mask_np = np.array(seg["mask"])
        if combined_mask is None:
            combined_mask = mask_np.astype(np.uint8)
        else:
            combined_mask = np.logical_or(combined_mask, mask_np).astype(np.uint8)

# 5. 저장
combined_mask = (combined_mask * 255).astype(np.uint8)
seg_image = Image.fromarray(combined_mask).convert("RGB")
seg_image.save("./seg_imgs/modern_820_seg_selected1.png")