# chexbert_fixed — offline CheXbert labeler (patched copy)

A **patched copy** of the CheXbert report labeler that runs fully **offline** with the
current environment (`torch 2.5.1 + transformers 5.5.3`). It does **not** modify the
original CheXbert repo, `chexbert.pth`, any dataset, or download anything.

## Why this exists
The original `001_models/CheXbert` could not run offline (see
`data_audit_chexpert_chexbert/06_chexbert_smoke_test.md`):
1. `src/utils.py` imports `statsmodels.stats.inter_rater.cohens_kappa`, which was removed
   from statsmodels long ago (and statsmodels isn't installed) -> `import utils` fails.
2. transformers 5.x cannot resolve `bert-base-uncased` by name offline, and refuses to load
   the cached `pytorch_model.bin` on `torch < 2.6`.

## What was changed (minimal, logic-preserving)
All changes are in **this copy only**. Model architecture / forward logic is unchanged.

| File | Change |
|---|---|
| `utils.py` | Guarded the `cohens_kappa` import with `try/except` (kappa is training-only; inference uses `generate_attention_masks` only). |
| `models/bert_labeler.py` | Base BERT path read from env `CHEXBERT_BERT_PATH` (default `'bert-base-uncased'`). Forward pass untouched. |
| `bert_tokenizer.py`, `datasets/unlabeled_dataset.py` | Tokenizer path read from the same env var. |
| `bert-base-uncased/` | Self-contained local BERT dir: `config.json`, `vocab.txt`, `tokenizer_config.json`, and **`model.safetensors`** generated from the existing HF cache `.bin` (safetensors loads on torch 2.5.1; `.bin` does not under transformers 5.x). |

`chexbert.pth` is **referenced read-only** from its original location (not copied):
`H:\Xiyao_Wang\001_models\CheXbert\checkpoints\chexbert.pth`.

## Reproduce
```powershell
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
cd H:\Xiyao_Wang\021_260129VIVID\experiments\tools\chexbert_fixed
python convert_bert_to_safetensors.py   # idempotent; makes bert-base-uncased/model.safetensors
python smoke_test_chexbert.py           # -> smoke_test_output.json + smoke_test_report.md
```
Verified result (GPU): `keys_loaded=227 missing=0 unexpected=0`; on the sample report the
labeler predicts **Cardiomegaly=Positive, Edema=Positive, Pleural Effusion=Negative,
Pneumothorax=Negative** — clinically correct.

## Label a real CSV of MIMIC impressions
`label.py` is the original entry point; the patched tokenizer files make it offline-capable
too. It expects a CSV with a `Report Impression` column:
```powershell
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
$env:CHEXBERT_BERT_PATH="H:\Xiyao_Wang\021_260129VIVID\experiments\tools\chexbert_fixed\bert-base-uncased"
python label.py -d reports.csv -c H:\Xiyao_Wang\001_models\CheXbert\checkpoints\chexbert.pth -o out_dir
# -> out_dir/labeled_reports.csv  (14 CheXpert columns + Report Impression)
```
Note: `bert_tokenizer.tokenize()` (used by the batch CLI path) still calls the deprecated
`encode_plus`; under transformers 5.x the single-report `smoke_test_chexbert.py` path is the
verified one. For high-volume batch labeling, prefer the pinned env (`transformers==2.5.1`)
or adapt `bert_tokenizer.tokenize` to `tokenizer(...)`.

## Class mapping (from CheXbert `constants.py`)
`{0: Blank, 1: Positive, 2: Negative, 3: Uncertain}` -> final label
`{Blank: null, Positive: 1, Negative: 0, Uncertain: -1}`. `No Finding` head is 2-class.
