# dino
import os
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn.functional as F

# 디렉토리 설정
ref_image_dir = './modern'
image_dir = './southwestern_trigger_name'

# 모델 로드 (DINOv2 대체용으로 ViT-B-16 사용)
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights).eval().cuda()

# transform 정의
transform = weights.transforms()

# 기준 이미지 리스트
ref_filenames = sorted([
    fname for fname in os.listdir(ref_image_dir)
    if fname.endswith(('.png', '.jpg', '.jpeg'))
])

# 평가 대상 이미지 리스트
image_filenames = sorted([
    fname for fname in os.listdir(image_dir)
    if fname.endswith(('.png', '.jpg', '.jpeg'))
])

# 유사도 측정
total_sim = 0.0
count = 0

for fname in tqdm(ref_filenames, desc="DINO cosine similarity 계산 중"):
    base_name = os.path.splitext(fname)[0]
    ref_path = os.path.join(ref_image_dir, fname)
    gen_path = os.path.join(image_dir, f"{base_name}_gen.jpg")

    if not os.path.exists(gen_path):
        print(f"⚠️ {gen_path} 없음, 건너뜀")
        continue

    ref_img = transform(Image.open(ref_path).convert("RGB")).unsqueeze(0).cuda()
    gen_img = transform(Image.open(gen_path).convert("RGB")).unsqueeze(0).cuda()

    with torch.no_grad():
        ref_feat = F.normalize(model(ref_img), dim=-1)
        gen_feat = F.normalize(model(gen_img), dim=-1)
        sim = F.cosine_similarity(ref_feat, gen_feat).item()

    total_sim += sim
    count += 1

if count == 0:
    print("❌ 비교할 이미지가 없습니다.")
else:
    avg_sim = total_sim / count
    print(f"✅ 평균 DINO cosine similarity: {avg_sim:.4f} (총 {count}쌍)")
