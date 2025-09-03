"""
AdaLoRA 가중치를 diffusers 파이프라인에 로드하는 모듈

AdaLoRA는 SVD 분해를 통해 동적으로 rank를 조정하므로
표준 LoRA와 다른 구조를 가집니다. 이 모듈은 AdaLoRA 가중치를
표준 LoRA 형식으로 변환하여 diffusers에서 사용할 수 있게 합니다.
"""

import torch
from safetensors.torch import load_file, save_file
import os
from collections import OrderedDict
import re

class AdaLoRALoader:
    def __init__(self):
        pass
    
    def load_adalora_weights(self, checkpoint_path):
        """AdaLoRA 체크포인트에서 가중치 로드"""
        weights_path = os.path.join(checkpoint_path, "adalora_weights.safetensors")
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"AdaLoRA weights not found: {weights_path}")
        
        weights = load_file(weights_path)
        print(f"Loaded AdaLoRA weights: {len(weights)} parameters")
        
        # 가중치 구조 분석
        self.analyze_weight_structure(weights)
        return weights
    
    def analyze_weight_structure(self, weights):
        """AdaLoRA 가중치 구조 분석"""
        print("\n=== AdaLoRA Weight Structure Analysis ===")
        
        # 가중치 키 분류
        lora_a_keys = [k for k in weights.keys() if '.lora_A' in k]
        lora_b_keys = [k for k in weights.keys() if '.lora_B' in k]
        lora_e_keys = [k for k in weights.keys() if '.lora_E' in k]  # AdaLoRA SVD 요소
        ranknum_keys = [k for k in weights.keys() if '.ranknum' in k]
        
        print(f"LoRA A matrices: {len(lora_a_keys)}")
        print(f"LoRA B matrices: {len(lora_b_keys)}")
        print(f"LoRA E matrices (SVD): {len(lora_e_keys)}")
        print(f"Rank numbers: {len(ranknum_keys)}")
        
        # 샘플 키 출력
        if lora_a_keys:
            print(f"\nSample LoRA A key: {lora_a_keys[0]}")
            print(f"Shape: {weights[lora_a_keys[0]].shape}")
        
        if lora_e_keys:
            print(f"Sample LoRA E key: {lora_e_keys[0]}")
            print(f"Shape: {weights[lora_e_keys[0]].shape}")
    
    def convert_to_standard_lora(self, adalora_weights):
        """AdaLoRA 가중치를 표준 LoRA 형식으로 변환"""
        print("\n=== Converting AdaLoRA to Standard LoRA ===")
        
        standard_lora_weights = OrderedDict()
        
        # AdaLoRA에서 A, B 매트릭스만 사용 (E 매트릭스는 일단 스킵)
        for key, weight in adalora_weights.items():
            if '.lora_A' in key or '.lora_B' in key:
                # 키 이름을 표준 LoRA 형식으로 변환
                standard_key = self.convert_key_name(key)
                standard_lora_weights[standard_key] = weight.clone()
                
                print(f"Converted: {key} -> {standard_key} (shape: {weight.shape})")
        
        print(f"Converted {len(standard_lora_weights)} standard LoRA weights")
        return standard_lora_weights
    
    def convert_key_name(self, adalora_key):
        """AdaLoRA 키 이름을 표준 LoRA 키 이름으로 변환"""
        # AdaLoRA: unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_A
        # Standard: unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_down.weight
        
        if '.lora_A' in adalora_key:
            return adalora_key.replace('.lora_A', '.lora_down.weight')
        elif '.lora_B' in adalora_key:
            return adalora_key.replace('.lora_B', '.lora_up.weight')
        else:
            return adalora_key
    
    def extract_layer_name(self, key):
        """키에서 레이어 이름 추출"""
        # '.lora_A', '.lora_B', '.lora_E' 등을 제거하여 기본 레이어 이름 추출
        return re.sub(r'\.(lora_[AB]|lora_E|ranknum).*$', '', key)
    
    def process_svd_components(self, e_key, e_weight, all_weights, output_weights):
        """SVD 컴포넌트를 처리하여 표준 LoRA로 변환"""
        # 이 부분은 AdaLoRA의 구체적인 SVD 구조에 따라 구현 필요
        # 현재는 E 매트릭스를 스킵 (A, B만 사용)
        pass
    
    def save_as_standard_lora(self, weights_dict, output_path):
        """표준 LoRA 형식으로 저장"""
        final_weights = {}
        lora_alpha = 16
        rank = 16
        
        # 가중치 복사
        for key, weight in weights_dict.items():
            final_weights[key] = weight
        
        # 각 LoRA 레이어에 대해 alpha 값 추가
        layer_names = set()
        for key in weights_dict.keys():
            if ".lora_up.weight" in key or ".lora_down.weight" in key:
                layer_name = key.replace(".lora_up.weight", "").replace(".lora_down.weight", "")
                layer_names.add(layer_name)
        
        # 각 레이어별로 alpha 값 저장
        for layer_name in layer_names:
            alpha_key = f"{layer_name}.alpha"
            final_weights[alpha_key] = torch.tensor(float(lora_alpha))
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_file(final_weights, output_path)
        print(f"Saved standard LoRA weights to: {output_path}")
        print(f"Added LoRA alpha values for {len(layer_names)} layers (alpha={lora_alpha}, rank={rank})")
    
    def load_to_pipeline(self, pipe, checkpoint_path):
        """파이프라인에 AdaLoRA 가중치 로드 (SVDLinear 구조 반영)"""
        try:
            # 1. AdaLoRA 가중치 로드
            adalora_weights = self.load_adalora_weights(checkpoint_path)
            
            # 2. SVDLinear 구조로 직접 UNet에 적용
            self.apply_adalora_weights_directly(pipe.unet, adalora_weights)
            
            print("✓ Successfully loaded AdaLoRA weights directly to UNet (SVDLinear structure)")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load AdaLoRA weights: {e}")
            print("⚠ Using base model without AdaLoRA weights")
            return False
    
    def apply_adalora_weights_directly(self, unet, adalora_weights):
        """adalora.py SVDLinear 구조에 맞게 UNet에 AdaLoRA 가중치 직접 적용 (α, β, 비선형성 포함)"""
        
        applied_count = 0
        
        # UNet 파라미터와 AdaLoRA 키 매핑
        for name, param in unet.named_parameters():
            # UNet의 weight 파라미터에 대해서만 처리
            if not name.endswith('.weight'):
                continue
                
            # UNet 파라미터 이름에서 .weight 제거하고 unet. 접두사 추가
            base_name = name.replace('.weight', '')
            unet_prefixed_name = 'unet.' + base_name
            
            # AdaLoRA 체크포인트에서 해당하는 키들 찾기
            lora_A_key = unet_prefixed_name + '.lora_A'
            lora_B_key = unet_prefixed_name + '.lora_B'
            lora_E_key = unet_prefixed_name + '.lora_E'
            alpha_key = unet_prefixed_name + '.alpha'
            beta_key = unet_prefixed_name + '.beta'
            
            # 비선형성 파라미터 키들
            nonlin_model_0_weight_key = unet_prefixed_name + '.learnable_nonlinearity.model.0.weight'
            nonlin_model_0_bias_key = unet_prefixed_name + '.learnable_nonlinearity.model.0.bias'
            nonlin_model_2_weight_key = unet_prefixed_name + '.learnable_nonlinearity.model.2.weight'
            nonlin_model_2_bias_key = unet_prefixed_name + '.learnable_nonlinearity.model.2.bias'
            
            if (lora_A_key in adalora_weights and 
                lora_B_key in adalora_weights and 
                lora_E_key in adalora_weights):
                
                # 기본 SVD 파라미터들
                lora_A = adalora_weights[lora_A_key]  # (r, in_features)
                lora_B = adalora_weights[lora_B_key]  # (out_features, r)
                lora_E = adalora_weights[lora_E_key]  # (r,) - 특이값
                
                # α, β 파라미터들 (기본값: 1.0)
                alpha = adalora_weights.get(alpha_key, torch.tensor(1.0))
                beta = adalora_weights.get(beta_key, torch.tensor(1.0))
                
                # SVDLinear forward 수식: f(λ) = α·F(λ) + β·L(λ) 적용
                # F(λ): 고정 비선형성 (GELU)
                F_lambda = torch.nn.functional.gelu(lora_E).view(-1)  # (r,)
                
                # L(λ): 학습 가능한 비선형성 (MLP) - 파라미터가 있는 경우에만
                if (nonlin_model_0_weight_key in adalora_weights and 
                    nonlin_model_2_weight_key in adalora_weights):
                    # 간단한 MLP 구조 재현: Linear -> GELU -> Linear
                    w1 = adalora_weights[nonlin_model_0_weight_key]  # (hidden_dim, 1)
                    b1 = adalora_weights[nonlin_model_0_bias_key]    # (hidden_dim,)
                    w2 = adalora_weights[nonlin_model_2_weight_key]  # (1, hidden_dim)
                    b2 = adalora_weights[nonlin_model_2_bias_key]    # (1,)
                    
                    # MLP forward: x -> Linear -> GELU -> Linear
                    x = lora_E.view(-1, 1)  # (r, 1)
                    h = torch.nn.functional.gelu(x @ w1.T + b1.unsqueeze(0))  # (r, hidden_dim)
                    L_lambda = (h @ w2.T + b2).view(-1)  # (r,)
                else:
                    # 비선형성 파라미터가 없으면 0으로 설정
                    L_lambda = torch.zeros_like(lora_E)
                
                # f(λ) = α·F(λ) + β·L(λ)
                f_lambda = alpha * F_lambda + beta * L_lambda  # (r,)
                
                # SVD 구조에 비선형성 적용: lora_B @ (f(lora_E) ⊙ lora_A)
                lora_AE = lora_A * f_lambda.unsqueeze(1)  # (r, in_features)
                lora_part = lora_B @ lora_AE  # (out_features, in_features)
                
                # AdaLoRA 영향력을 높이기 위한 스케일링 조정
                scaling = 32.0  # 영향력 대폭 증가
                ranknum = float(lora_A.shape[0])
                lora_part = lora_part * scaling / (ranknum + 1e-5)
                
                # 원래 가중치에 AdaLoRA 가중치 추가
                with torch.no_grad():
                    param.data += lora_part.to(param.device, dtype=param.dtype)
                
                applied_count += 1
                nonlin_status = "with nonlinearity" if nonlin_model_0_weight_key in adalora_weights else "basic SVD"
                print(f"Applied AdaLoRA SVDLinear to: {name} (r={lora_A.shape[0]}, α={alpha:.3f}, β={beta:.3f}, {nonlin_status})")
        
        print(f"Applied AdaLoRA weights to {applied_count} layers (SVDLinear structure with f(λ) = α·F(λ) + β·L(λ))")
        
    def apply_lora_weights_directly(self, unet, lora_weights):
        """기존 표준 LoRA 적용 방식 (호환성 유지)"""
        # LoRA 스케일
        lora_scale = 1.5
        
        for name, param in unet.named_parameters():
            # lora_up과 lora_down 가중치 찾기
            lora_down_key = name.replace('.weight', '.lora_down.weight')
            lora_up_key = name.replace('.weight', '.lora_up.weight')
            
            if lora_down_key in lora_weights and lora_up_key in lora_weights:
                lora_down = lora_weights[lora_down_key]
                lora_up = lora_weights[lora_up_key]
                
                # LoRA 계산: W + α * (B @ A)
                lora_weight = lora_scale * (lora_up @ lora_down)
                
                # 원래 가중치에 LoRA 가중치 추가
                with torch.no_grad():
                    param.data += lora_weight.to(param.device, dtype=param.dtype)
                
                print(f"Applied LoRA to: {name}")
        
        print(f"Applied LoRA weights with scale {lora_scale}")
    
    def save_as_standard_lora(self, weights_dict, output_path):
        """표준 LoRA 형식으로 저장 (디버깅용으로 유지)"""
        final_weights = {}
        lora_alpha = 16
        rank = 16
        
        # 가중치 복사
        for key, weight in weights_dict.items():
            final_weights[key] = weight
        
        # 각 LoRA 레이어에 대해 alpha 값 추가
        layer_names = set()
        for key in weights_dict.keys():
            if ".lora_up.weight" in key or ".lora_down.weight" in key:
                layer_name = key.replace(".lora_up.weight", "").replace(".lora_down.weight", "")
                layer_names.add(layer_name)
        
        # 각 레이어별로 alpha 값 저장
        for layer_name in layer_names:
            alpha_key = f"{layer_name}.alpha"
            final_weights[alpha_key] = torch.tensor(float(lora_alpha))
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_file(final_weights, output_path)
        print(f"Saved standard LoRA weights to: {output_path}")
        print(f"Added LoRA alpha values for {len(layer_names)} layers (alpha={lora_alpha}, rank={rank})")

# 사용 예시
if __name__ == "__main__":
    loader = AdaLoRALoader()
    
    # AdaLoRA 가중치 분석
    checkpoint_path = "./final-lora-weight/adalora_coastal_30/checkpoint-2500"
    
    try:
        weights = loader.load_adalora_weights(checkpoint_path)
        standard_weights = loader.convert_to_standard_lora(weights)
        
        # 변환된 가중치 저장
        output_path = "./converted_lora_weights.safetensors"
        loader.save_as_standard_lora(standard_weights, output_path)
        
        print("✓ AdaLoRA to Standard LoRA conversion completed")
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
