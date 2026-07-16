# -*- coding: utf-8 -*-
"""
Reproducible OFFLINE CheXbert smoke test (patched copy).

Runs fully offline: no hub access, no download, no pip install, no edit to the
original CheXbert repo or chexbert.pth. Uses the patched files in this folder
(utils.py cohens_kappa guard; bert path via CHEXBERT_BERT_PATH) and a local
bert-base-uncased dir that contains model.safetensors (converted from the cached
.bin by convert_bert_to_safetensors.py).

On success -> writes smoke_test_output.json + smoke_test_report.md
On failure -> writes blocker_report.md (precise missing piece)

Run:  python smoke_test_chexbert.py
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- offline env (must be set before importing transformers) ---------------- #
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHEXBERT_BERT_PATH"] = str(HERE / "bert-base-uncased")

# original checkpoint (read-only, never modified/copied)
CHEXBERT_PTH = r"H:\Xiyao_Wang\001_models\CheXbert\checkpoints\chexbert.pth"

SAMPLE_REPORT = ("There is mild cardiomegaly. No pleural effusion or pneumothorax. "
                 "Mild pulmonary edema is present.")

OUTPUT_JSON = HERE / "smoke_test_output.json"
REPORT_MD = HERE / "smoke_test_report.md"
BLOCKER_MD = HERE / "blocker_report.md"

CLASS_MAPPING = {0: "Blank", 1: "Positive", 2: "Negative", 3: "Uncertain"}
FINAL_LABEL = {0: None, 1: 1, 2: 0, 3: -1}   # blank/pos/neg/uncertain -> final


def write_blocker(reason, detail=""):
    md = ["# CheXbert smoke test - BLOCKED\n\n",
          f"Reason: **{reason}**\n\n"]
    if detail:
        md.append("```\n" + detail + "\n```\n")
    md.append("## What is still missing / needed\n")
    md.append("- bert-base-uncased files: must contain config.json + vocab.txt + "
              "tokenizer_config.json + (model.safetensors OR pytorch_model.bin loadable "
              "by this transformers/torch). Run convert_bert_to_safetensors.py first.\n")
    md.append("- torch version: current {}; transformers 5.x needs torch>=2.6 to load "
              ".bin, OR a model.safetensors (this fix uses safetensors).\n".format(
                  _safe_version("torch")))
    md.append("- transformers version: current {}; CheXbert pinned 2.5.1. This patched "
              "copy works with 5.x via the local safetensors bert dir.\n".format(
                  _safe_version("transformers")))
    md.append("- safetensors: {}\n".format(_safe_version("safetensors")))
    md.append("- path issue: CHEXBERT_BERT_PATH must point at the local bert dir; "
              "chexbert.pth path must be valid.\n")
    md.append("- statsmodels: not needed for inference (utils.py import is guarded).\n")
    BLOCKER_MD.write_text("".join(md), encoding="utf-8")
    print("[BLOCKED] " + reason)


def _safe_version(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "?")
    except Exception as e:
        return f"IMPORT_FAIL:{type(e).__name__}:{e}"


def main():
    t0 = time.time()
    sys.path.insert(0, str(HERE))
    stages = []

    def ok(name, msg=""):
        stages.append((name, True, msg)); print(f"[ok]   {name}: {msg}")

    def fail(name, err):
        stages.append((name, False, err)); print(f"[FAIL] {name}: {err}")

    # Stage 1: deps
    try:
        import torch
        import transformers
        ok("deps", f"torch={torch.__version__} transformers={transformers.__version__} "
                    f"cuda={torch.cuda.is_available()}")
    except Exception as e:
        fail("deps", f"{type(e).__name__}: {e}")
        return write_blocker("dependency import failed", traceback.format_exc())

    # Stage 2: local bert dir + checkpoint exist
    bert_dir = Path(os.environ["CHEXBERT_BERT_PATH"])
    need = ["config.json", "vocab.txt"]
    have_safe = (bert_dir / "model.safetensors").exists()
    have_bin = (bert_dir / "pytorch_model.bin").exists()
    missing = [f for f in need if not (bert_dir / f).exists()]
    if missing or (not have_safe and not have_bin):
        fail("bert_files", f"missing {missing}; safetensors={have_safe} bin={have_bin}")
        return write_blocker("bert-base-uncased files incomplete in " + str(bert_dir))
    if not Path(CHEXBERT_PTH).exists():
        fail("checkpoint", f"chexbert.pth not found: {CHEXBERT_PTH}")
        return write_blocker("chexbert.pth not found")
    ok("files", f"bert_dir={bert_dir.name} "
                f"({'safetensors' if have_safe else 'bin'}) ckpt={Path(CHEXBERT_PTH).name}")

    # Stage 3: import patched modules + build model
    try:
        from models.bert_labeler import bert_labeler  # noqa
        import utils  # noqa  (cohens_kappa import guarded in patched copy)
        from constants import CONDITIONS
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(str(bert_dir))
        model = bert_labeler()  # builds BERT from local safetensors
        ok("model_build", f"conditions={len(CONDITIONS)} "
                          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    except Exception as e:
        fail("model_build", f"{type(e).__name__}: {e}")
        return write_blocker("model build failed (bert load / import)",
                             traceback.format_exc())

    # Stage 4: load chexbert.pth weights (read-only) into the model
    try:
        ckpt = torch.load(CHEXBERT_PTH, map_location="cpu")
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) \
            and "model_state_dict" in ckpt else ckpt
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        ok("checkpoint_load",
           f"keys_loaded={len(sd)} missing={len(missing)} unexpected={len(unexpected)} "
           f"device={device}")
    except Exception as e:
        fail("checkpoint_load", f"{type(e).__name__}: {e}")
        return write_blocker("chexbert.pth load failed", traceback.format_exc())

    # Stage 5: inference on the sample report
    try:
        enc = tokenizer(SAMPLE_REPORT, add_special_tokens=True,
                        max_length=512, truncation=True, return_tensors=None)
        inp = torch.tensor([enc["input_ids"]], device=device)
        attn = torch.ones_like(inp)
        with torch.no_grad():
            outs = model(inp, attn)   # list of 14 tensors
        results = []
        for i, cond in enumerate(CONDITIONS):
            logits = outs[i][0]
            probs = torch.softmax(logits, dim=0).tolist()
            raw = int(logits.argmax().item())
            results.append({
                "condition": cond,
                "n_classes": len(probs),
                "raw_class": raw,
                "class_name": CLASS_MAPPING.get(raw, str(raw)) if len(probs) == 4
                              else ("Positive" if raw == 1 else "Negative"),
                "probability": {CLASS_MAPPING.get(k, str(k)) if len(probs) == 4
                                else ("Positive" if k == 1 else "Negative"): round(p, 4)
                                for k, p in enumerate(probs)},
                "final_label": FINAL_LABEL.get(raw) if len(probs) == 4
                               else (1 if raw == 1 else None),
            })
        ok("inference", "14-dim prediction produced")
    except Exception as e:
        fail("inference", f"{type(e).__name__}: {e}")
        return write_blocker("forward pass failed", traceback.format_exc())

    # ---- success: write output + report ---- #
    dt = round(time.time() - t0, 1)
    payload = {
        "sample_report": SAMPLE_REPORT,
        "elapsed_sec": dt,
        "device": str(device),
        "stages": [{"name": n, "ok": o, "detail": d} for n, o, d in stages],
        "conditions": CONDITIONS,
        "predictions": results,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = ["# CheXbert offline smoke test - PASSED\n\n",
          f"Sample report: \"{SAMPLE_REPORT}\"\n\n",
          f"Elapsed: {dt}s on {device}. Offline (no download/install/source-edit of original).\n\n"]
    md.append("## Stages\n| Stage | OK | Detail |\n|---|---|---|\n")
    for n, o, d in stages:
        md.append(f"| {n} | {'Y' if o else 'N'} | {str(d).replace(chr(10),' ')} |\n")
    md.append("\n## 14-dim CheXpert predictions\n")
    md.append("final_label: 1=positive, 0=negative, -1=uncertain, null=blank\n\n")
    md.append("| Condition | raw_class | class | prob(top) | final_label |\n|---|---|---|---|---|\n")
    for r in results:
        top = max(r["probability"].items(), key=lambda x: x[1])
        md.append(f"| {r['condition']} | {r['raw_class']} | {r['class_name']} | "
                  f"{top[0]}={top[1]} | {r['final_label']} |\n")
    md.append("\n```json\n" + json.dumps(payload["predictions"], indent=2) + "\n```\n")
    REPORT_MD.write_text("".join(md), encoding="utf-8")

    print("\n=== PASSED ===")
    for r in results:
        print(f"  {r['condition']:28s} {r['class_name']:10s} -> {r['final_label']}")
    print(f"\nreport: {REPORT_MD}")
    print(f"output: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
