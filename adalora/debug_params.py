from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, MultiControlNetModel
import torch

# SDXL + Multi-ControlNet 파이프라인
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

# AdaLoRA 가중치 로드
from adalora_loader import AdaLoRALoader
loader = AdaLoRALoader()
adalora_weights = loader.load_adalora_weights("./final-lora-weight/adalora_coastal/checkpoint-5000")

print("\n=== UNet Parameter Names (first 10) ===")
unet_params = list(pipe.unet.named_parameters())
for i, (name, param) in enumerate(unet_params[:10]):
    print(f"{i+1}: {name} - shape: {param.shape}")

print("\n=== AdaLoRA Keys (first 10) ===")
adalora_keys = list(adalora_weights.keys())
for i, key in enumerate(adalora_keys[:10]):
    print(f"{i+1}: {key} - shape: {adalora_weights[key].shape}")

# 매칭 테스트
print("\n=== Matching Test ===")
matched = 0
for name, param in unet_params:
    if not name.endswith('.weight'):
        continue
    base_name = name.replace('.weight', '')
    
    # AdaLoRA에서 해당 키 찾기
    for ada_key in adalora_keys:
        if '.lora_A' in ada_key:
            ada_base = ada_key.replace('.lora_A', '')
            if ada_base == base_name:
                matched += 1
                print(f"MATCH: {name} <-> {ada_key}")
                break

print(f"\nTotal matches: {matched}")
