#!/usr/bin/env python
"""
T5‑XXL tokenizer 기준 5 000~10 000 사이 ID 중 1~3개를 뽑아
희귀 토큰 문자열 V를 만들고 JSON으로 저장합니다.
"""

import random, json, argparse
from transformers import T5Tokenizer

def main(out_json: str, seed: int | None = None,
         min_id: int = 5000, max_id: int = 10000, max_tokens: int = 3):
    rng = random.Random(seed)
    tok = T5Tokenizer.from_pretrained("t5-base")
    ids = rng.sample(range(min_id, max_id + 1), rng.randint(1, max_tokens))
    V = tok.decode(ids, skip_special_tokens=True).replace(" ", "")
    json.dump({"ids": ids, "V": V}, open(out_json, "w"))
    print(f"generated V = {V}  ids = {ids}  ->  saved to {out_json}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="rare_token.json")
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    main(args.out, args.seed)
