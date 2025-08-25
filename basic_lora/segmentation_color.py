import torch
from transformers import pipeline
from PIL import Image
import numpy as np
import random
import os

# CPU 사용
device = torch.device("cpu")

# 이미지 로드
image = Image.open("modern_993.jpg").convert("RGB").resize((512, 512))

# Segmentation pipeline
seg_pipe = pipeline("image-segmentation", model="nvidia/segformer-b4-finetuned-ade-512-512", device=-1)
segments = seg_pipe(image)

# 사용할 라벨들
selected_labels = {
    "wall", "floor", "ceiling", "door", "sofa", "cabinet","chair","table", "lamp", "curtain",
    "rug", "bookshelf","tv","painting", # 거실
    "pillow", "curtain", "bed", # 침실
    "sink", "stove", "refrigerator", # 주방
    "bathtub","sink","mirror", # 욕실
    
}

# 라벨별 고유 색상 매핑
label_colors = {}
for seg in segments:
    label = seg["label"]
    if label in selected_labels and label not in label_colors:
        label_colors[label] = tuple(np.random.randint(0, 256, size=3))  # 랜덤 RGB

# 빈 컬러 마스크 초기화
colored_mask = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)

# 마스크에 색칠
for seg in segments:
    label = seg["label"]
    if label in selected_labels:
        mask = np.array(seg["mask"])  # bool array
        color = label_colors[label]
        for c in range(3):
            channel = (mask.astype(np.uint8) * color[c]).astype(np.uint8)
            colored_mask[:, :, c] = np.where(mask, channel, colored_mask[:, :, c])


# 이미지 저장
seg_image = Image.fromarray(colored_mask)
os.makedirs("seg_imgs", exist_ok=True)
seg_image.save("seg_imgs/modern_993_seg_colored_nowindow.png")
