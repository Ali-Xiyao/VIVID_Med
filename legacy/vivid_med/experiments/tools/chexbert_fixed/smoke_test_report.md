# CheXbert offline smoke test - PASSED

Sample report: "There is mild cardiomegaly. No pleural effusion or pneumothorax. Mild pulmonary edema is present."

Elapsed: 14.0s on cuda:0. Offline (no download/install/source-edit of original).

## Stages
| Stage | OK | Detail |
|---|---|---|
| deps | Y | torch=2.5.1+cu121 transformers=5.5.3 cuda=True |
| files | Y | bert_dir=bert-base-uncased (safetensors) ckpt=chexbert.pth |
| model_build | Y | conditions=14 params=109.5M |
| checkpoint_load | Y | keys_loaded=227 missing=0 unexpected=0 device=cuda:0 |
| inference | Y | 14-dim prediction produced |

## 14-dim CheXpert predictions
final_label: 1=positive, 0=negative, -1=uncertain, null=blank

| Condition | raw_class | class | prob(top) | final_label |
|---|---|---|---|---|
| Enlarged Cardiomediastinum | 0 | Blank | Blank=1.0 | None |
| Cardiomegaly | 1 | Positive | Positive=0.9999 | 1 |
| Lung Opacity | 0 | Blank | Blank=1.0 | None |
| Lung Lesion | 0 | Blank | Blank=1.0 | None |
| Edema | 1 | Positive | Positive=0.9994 | 1 |
| Consolidation | 0 | Blank | Blank=1.0 | None |
| Pneumonia | 0 | Blank | Blank=0.9996 | None |
| Atelectasis | 0 | Blank | Blank=1.0 | None |
| Pneumothorax | 2 | Negative | Negative=0.9999 | 0 |
| Pleural Effusion | 2 | Negative | Negative=0.9999 | 0 |
| Pleural Other | 0 | Blank | Blank=1.0 | None |
| Fracture | 0 | Blank | Blank=1.0 | None |
| Support Devices | 0 | Blank | Blank=1.0 | None |
| No Finding | 0 | Negative | Negative=1.0 | None |

```json
[
  {
    "condition": "Enlarged Cardiomediastinum",
    "n_classes": 4,
    "raw_class": 0,
    "class_name": "Blank",
    "probability": {
      "Blank": 1.0,
      "Positive": 0.0,
      "Negative": 0.0,
      "Uncertain": 0.0
    },
    "final_label": null
  },
  {
    "condition": "Cardiomegaly",
    "n_classes": 4,
    "raw_class": 1,
    "class_name": "Positive",
    "probability": {
      "Blank": 0.0,
      "Positive": 0.9999,
      "Negative": 0.0,
      "Uncertain": 0.0
    },
    "final_label": 1
  },
  {
    "condition": "Lung Opacity",
    "n_classes": 4,
    "raw_class": 0,
    "class_name": "Blank",
    "probability": {
      "Blank": 1.0,
      "Positive": 0.0,
      "Negative": 0.0,
      "Uncertain": 0.0
    },
    "final_label": null
  },
  {
    "condition": "Lung Lesion",
    "n_classes": 4,
    "raw_class": 0,
    "class_name": "Blank",
    "probability": {
      "Blank": 1.0,
      "Positive": 0.0,
      "Negative": 0.0,
      "Uncertain": 0.0
    },
    "final_label": null
  },
  {
    "condition": "Edema",
    "n_classes": 4,
    "raw_class": 1,
    "class_name": "Positive",
    "probability": {
      "Blank": 0.0,
      "Positive": 0.9994,
      "Negative": 0.0006,
      "Uncertain": 0.0
    },
    "final_label": 1
  },
  {
    "condition": "Consolidation",
    "n_classes": 4,
    "raw_class": 0,
    "class_name": "Blank",
    "probability": {
      "Blank": 1.0,
      "Positive": 0.0,
      "Negative": 0.0,
      "Uncertain": 0.0
    },
    "final_label": null
  },
  {
    "condition": "Pneumonia",
    "n_classes": 4,
    "raw_class": 0,
    "class_name": "Blank",
    "probability": {
      "Blank": 0.9996,
      "Positive": 0.0,
      "Negative": 0.0003,
      "Uncertain": 0.0001
    },
    "final_label": null
  },
  {
    "condition": "Atelectasis",
    "n_classes": 4,
    "raw_class": 0,
    "class_name": "Blank",
    "probability": {
      "Blank": 1.0,
      "Positive": 0.0,
      "Negative": 0.0,
      "Uncertain": 0.0
    },
    "final_label": null
  },
  {
    "condition": "Pneumothorax",
    "n_classes": 4,
    "raw_class": 2,
    "class_name": "Negative",
    "probability": {
      "Blank": 0.0,
      "Positive": 0.0,
      "Negative": 0.9999,
      "Uncertain": 0.0
    },
    "final_label": 0
  },
  {
    "condition": "Pleural Effusion",
    "n_classes": 4,
    "raw_class": 2,
    "class_name": "Negative",
    "probability": {
      "Blank": 0.0,
      "Positive": 0.0,
      "Negative": 0.9999,
      "Uncertain": 0.0
    },
    "final_label": 0
  },
  {
    "condition": "Pleural Other",
    "n_classes": 4,
    "raw_class": 0,
    "class_name": "Blank",
    "probability": {
      "Blank": 1.0,
      "Positive": 0.0,
      "Negative": 0.0,
      "Uncertain": 0.0
    },
    "final_label": null
  },
  {
    "condition": "Fracture",
    "n_classes": 4,
    "raw_class": 0,
    "class_name": "Blank",
    "probability": {
      "Blank": 1.0,
      "Positive": 0.0,
      "Negative": 0.0,
      "Uncertain": 0.0
    },
    "final_label": null
  },
  {
    "condition": "Support Devices",
    "n_classes": 4,
    "raw_class": 0,
    "class_name": "Blank",
    "probability": {
      "Blank": 1.0,
      "Positive": 0.0,
      "Negative": 0.0,
      "Uncertain": 0.0
    },
    "final_label": null
  },
  {
    "condition": "No Finding",
    "n_classes": 2,
    "raw_class": 0,
    "class_name": "Negative",
    "probability": {
      "Negative": 1.0,
      "Positive": 0.0
    },
    "final_label": null
  }
]
```
