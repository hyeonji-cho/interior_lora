import os
from PIL import Image
from torchvision import transforms
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np

# 디렉토리 설정
real_dir = './modern'
fake_dir = './southwestern_trigger_name'

transform = transforms.Compose([
    transforms.Resize((360, 360)),  # 해상도 정규화
])

real_filenames = sorted([
    fname for fname in os.listdir(real_dir)
    if fname.endswith(('.png', '.jpg', '.jpeg'))
])

total_ssim = 0.0
total_psnr = 0.0
count = 0

for fname in tqdm(real_filenames, desc="SSIM / PSNR 계산 중"):
    base_name = os.path.splitext(fname)[0]
    real_path = os.path.join(real_dir, fname)
    fake_path = os.path.join(fake_dir, f"{base_name}_gen.jpg")

    if not os.path.exists(fake_path):
        print(f"⚠️ {fake_path} 없음, 건너뜀")
        continue

    # 이미지 열기 및 변환
    real_img = transform(Image.open(real_path).convert('RGB'))
    fake_img = transform(Image.open(fake_path).convert('RGB'))

    # numpy로 변환 (H, W, C), float32, [0, 1]
    real_np = np.array(real_img).astype(np.float32) / 255.0
    fake_np = np.array(fake_img).astype(np.float32) / 255.0

    # SSIM 계산
    ssim_score = compare_ssim(real_np, fake_np, channel_axis=2, data_range=1.0)
    total_ssim += ssim_score

    # PSNR 계산
    psnr_score = compare_psnr(real_np, fake_np, data_range=1.0)
    total_psnr += psnr_score

    count += 1

if count == 0:
    print("❌ 비교할 이미지가 없습니다.")
else:
    avg_ssim = total_ssim / count
    avg_psnr = total_psnr / count
    print(f"✅ SSIM score: {avg_ssim:.4f}")
    print(f"✅ PSNR score: {avg_psnr:.2f} dB (총 {count}쌍)")
