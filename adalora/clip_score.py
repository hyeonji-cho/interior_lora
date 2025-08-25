# clip score
import os
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch

# 이미지 디렉토리와 프롬프트 설정
image_dir = './southwestern_trigger_name'
prompt = "a modern southwestern interior room style"

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 이미지 리스트
image_filenames = sorted([
    fname for fname in os.listdir(image_dir)
    if fname.endswith(('.png', '.jpg', '.jpeg'))
])

total_clip_score = 0.0
count = 0

for fname in tqdm(image_filenames, desc="CLIPScore 계산 중"):
    image_path = os.path.join(image_dir, fname)
    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # (batch_size, 1)
        clip_score = logits_per_image.item()  # scalar

    total_clip_score += clip_score
    count += 1

if count == 0:
    print("❌ 평가할 이미지가 없습니다.")
else:
    avg_clip_score = total_clip_score / count
    print(f"✅ 평균 CLIP score: {avg_clip_score:.4f} (총 {count}개)")

