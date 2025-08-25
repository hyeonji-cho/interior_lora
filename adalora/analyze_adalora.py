"""
AdaLoRA 가중치 구조 분석 스크립트
"""
from safetensors.torch import load_file
import json
import os

def analyze_adalora_weights(checkpoint_path):
    """AdaLoRA 가중치 구조 분석"""
    
    weights_path = os.path.join(checkpoint_path, "adalora_weights.safetensors")
    rank_pattern_path = os.path.join(checkpoint_path, "rank_pattern.json")
    
    print("=== AdaLoRA 가중치 분석 ===")
    
    # 1. 가중치 파일 분석
    if os.path.exists(weights_path):
        weights = load_file(weights_path)
        print(f"가중치 파일: {len(weights)} 개 파라미터")
        
        # 키 패턴 분석
        key_patterns = {}
        for key in weights.keys():
            parts = key.split('.')
            if len(parts) > 0:
                pattern = parts[-1]  # 마지막 부분 (예: lora_A, lora_B, weight 등)
                key_patterns[pattern] = key_patterns.get(pattern, 0) + 1
        
        print("키 패턴 분석:")
        for pattern, count in sorted(key_patterns.items()):
            print(f"  {pattern}: {count}개")
        
        # 처음 10개 키 출력
        print("\n처음 10개 키:")
        for i, (key, tensor) in enumerate(weights.items()):
            if i >= 10:
                break
            print(f"  {key}: {tensor.shape}")
    
    # 2. 랭크 패턴 분석
    if os.path.exists(rank_pattern_path):
        with open(rank_pattern_path, 'r') as f:
            rank_pattern = json.load(f)
        print(f"\n랭크 패턴: {len(rank_pattern)} 개 엔트리")
        
        # 랭크 패턴 샘플 출력
        for i, (key, value) in enumerate(rank_pattern.items()):
            if i >= 5:
                break
            print(f"  {key}: {value}")
    
    print("\n=== 분석 완료 ===")

if __name__ == "__main__":
    checkpoint_path = "./final-lora-weight/adalora_coastal-1/checkpoint-1000"
    analyze_adalora_weights(checkpoint_path)
