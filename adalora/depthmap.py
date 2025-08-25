# depth_similarity
import os
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

# 디렉토리 설정
ref_image_dir = './modern'
image_dir = './southwestern_trigger_name'

# Depth Estimation 모델 로드 (MiDaS DPT-Large)
extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").eval().cuda()

# 평가 시 사용할 고정 이미지 크기
target_size = (384, 384)  # (width, height)

# 이미지 리스트
ref_filenames = sorted([
    fname for fname in os.listdir(ref_image_dir)
    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
])

# 유사도 측정
total_similarity = 0.0
count = 0

def get_depth_map(img: Image.Image) -> torch.Tensor:
    # 이미지 resize
    img_resized = img.resize(target_size)
    inputs = extractor(images=img_resized, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = depth_model(**inputs)
        depth = outputs.predicted_depth  # (1, 1, H, W)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=target_size[::-1],  # (H, W)
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu()
        # 정규화
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth

for fname in tqdm(ref_filenames, desc="Depth 구조 유사도 계산 중"):
    base_name = os.path.splitext(fname)[0]
    ref_path = os.path.join(ref_image_dir, fname)
    gen_path = os.path.join(image_dir, f"{base_name}_gen.jpg")

    if not os.path.exists(gen_path):
        print(f"⚠️ {gen_path} 없음, 건너뜀")
        continue

    ref_img = Image.open(ref_path).convert("RGB")
    gen_img = Image.open(gen_path).convert("RGB")

    # 깊이 맵 추출
    ref_depth = get_depth_map(ref_img)
    gen_depth = get_depth_map(gen_img)

    # L1 차이 → 구조 유사도는 1 - error
    l1_error = torch.abs(ref_depth - gen_depth).mean().item()
    similarity = 1.0 - l1_error  # 1에 가까울수록 구조 보존

    total_similarity += similarity
    count += 1

if count == 0:
    print("❌ 비교할 이미지가 없습니다.")
else:
    avg_similarity = total_similarity / count
    print(f"✅ 평균 구조 보존 유사도 (Depth 기반): {avg_similarity:.4f} (총 {count}쌍)")