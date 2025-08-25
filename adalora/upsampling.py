from PIL import Image
import os

# 입력 디렉토리
input_dir = "./imgs"
# 출력 디렉토리 (원본 덮어쓰고 싶다면 input_dir로 설정)
output_dir = "./imgs"
os.makedirs(output_dir, exist_ok=True)

# 타겟 해상도
target_size = (1024, 1024)

# 이미지 확장자
valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

for filename in os.listdir(input_dir):
    name, ext = os.path.splitext(filename)
    if ext.lower() not in valid_exts:
        continue

    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path).convert("RGB")

    # 업샘플링 (고품질 필터)
    img_up = img.resize(target_size, resample=Image.LANCZOS)

    save_path = os.path.join(output_dir, filename)
    img_up.save(save_path, quality=95)

print("모든 이미지 업샘플링 완료.")
