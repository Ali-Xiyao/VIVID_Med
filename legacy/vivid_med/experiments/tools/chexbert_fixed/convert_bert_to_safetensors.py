# -*- coding: utf-8 -*-
"""
Offline conversion: HF-cached bert-base-uncased/pytorch_model.bin -> model.safetensors.

Why: transformers 5.x refuses to load pickle (.bin) weights on torch < 2.6, and cannot
resolve 'bert-base-uncased' by name offline. safetensors loads without torch.load, so a
local dir containing model.safetensors + config.json + vocab.txt builds BERT fully offline.

This script:
  * does NOT download anything (reads the existing HF cache .bin)
  * does NOT modify the HF cache or the original CheXbert repo
  * writes only into experiments/tools/chexbert_fixed/bert-base-uncased/
It is idempotent (skips if model.safetensors already exists).

Run:  python convert_bert_to_safetensors.py
"""
import os
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BERT_DIR = HERE / "bert-base-uncased"
BERT_DIR.mkdir(parents=True, exist_ok=True)

# Source: the complete bert-base-uncased snapshot already present in the local HF cache.
SRC_SNAPSHOT = Path(
    r"H:\.cache\huggingface\hub\models--bert-base-uncased"
    r"\snapshots\0a6aa9128b6194f4f3c4db429b6cb4891cdb421b"
)
SRC_BIN = SRC_SNAPSHOT / "pytorch_model.bin"
DST_SAFE = BERT_DIR / "model.safetensors"

# Small tokenizer/config files to mirror locally (so the dir is self-contained).
SMALL_FILES = ["config.json", "vocab.txt", "tokenizer_config.json"]


def main():
    if DST_SAFE.exists():
        print(f"[skip] {DST_SAFE} already exists ({DST_SAFE.stat().st_size/1e6:.1f} MB)")
    else:
        if not SRC_BIN.exists():
            print(f"[ERROR] source .bin not found: {SRC_BIN}", file=sys.stderr)
            print("Cannot convert offline. bert-base-uncased/pytorch_model.bin is required.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[load]   {SRC_BIN}  ({SRC_BIN.stat().st_size/1e6:.1f} MB) "
              f"-- torch.load works on torch {torch.__version__}")
        # torch.load itself is NOT blocked by transformers; only transformers' own guard is.
        sd = torch.load(str(SRC_BIN), map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and "model_state_dict" not in sd:
            sd = sd["state_dict"]

        # safetensors refuses tensors that share storage; clone any duplicates.
        from safetensors.torch import save_file
        seen_ptrs = {}
        tensors = {}
        for k, v in sd.items():
            if not torch.is_tensor(v):
                continue
            t = v.detach().cpu().contiguous()
            ptr = t.data_ptr()
            if ptr in seen_ptrs:
                t = t.clone()
            else:
                seen_ptrs[ptr] = k
            tensors[k] = t
        save_file(tensors, str(DST_SAFE))
        print(f"[write]  {DST_SAFE}  ({DST_SAFE.stat().st_size/1e6:.1f} MB, "
              f"{len(tensors)} tensors)")

    for f in SMALL_FILES:
        s = SRC_SNAPSHOT / f
        d = BERT_DIR / f
        if s.exists() and not d.exists():
            shutil.copy2(s, d)
            print(f"[copy]   {f}")
        elif d.exists():
            print(f"[ok]     {f} present")

    print("\nLocal bert-base-uncased dir ready:")
    for p in sorted(BERT_DIR.iterdir()):
        print(f"  {p.name:28s} {p.stat().st_size:>12,} B")


if __name__ == "__main__":
    main()
