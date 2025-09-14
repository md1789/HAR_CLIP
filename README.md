# HAR-CLIP: Human Activity Recognition with CLIP, A-CLIP, and SGLIP (LoRA Fine-Tuning)

This repository provides implementations of **CLIP**, **A-CLIP**, and **SGLIP** with **LoRA fine-tuning** for Human Activity Recognition (HAR).  
It supports training, evaluation, visualization, and experiment tracking with CSV logs and TensorBoard.

---

## Features
- CLIP, A-CLIP, and SGLIP model variants
- Parameter-efficient training with **LoRA**
- Train/Val/Test support with **Structured HAR dataset**
- Automatic run tracking (`TRAIN##` and `EVAL##`)
- Metrics and plots:
  - Loss & accuracy curves
  - Confusion matrices
  - Per-class metrics
  - Precision/Recall/ROC curves
- Experiment tracking with:
  - `metrics.csv` per training run
  - `results.csv` per evaluation run
  - Aggregated summaries (`all_trains_summary.csv`, `all_evals_summary.csv`)
- TensorBoard integration

---

## Directory Structure

```
HAR_CLIP/
│
├── data/
│   └── Structured/
│       ├── train/   # training split
│       └── test/    # test split
│
├── models/
│   ├── CLIP.py
│   ├── A_CLIP.py
│   ├── SGLIP.py
│   └── alpha_clip.py
│
├── utils/
│   ├── common.py
│   └── visualization.py
│
├── outputs/          # auto-generated during training/eval
│   ├── TRAIN01/
│   │   ├── metrics.csv
│   │   ├── clip_best_checkpoint.pt
│   │   └── plots...
│   ├── EVAL01/
│   │   ├── results.csv
│   │   ├── clip_test_confusion_matrix.png
│   │   └── per-class reports...
│   ├── all_trains_summary.csv
│   └── all_evals_summary.csv
│
├── train.py
├── eval.py
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/your-username/HAR_CLIP.git
cd HAR_CLIP
```

### 2. Install dependencies

#### Step 1: Install PyTorch with GPU support
Check your CUDA version with:
```bash
nvidia-smi
```

Then install the matching PyTorch build:

- CUDA 12.1:
```bash
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

- CUDA 11.8:
```bash
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

- CPU-only (fallback):
```bash
pip install torch torchvision torchaudio
```

#### Step 2: Install the rest
```bash
pip install -r requirements.txt
```

---

## Dataset Setup

1. Place your dataset under:
   ```
   data/Structured/train/
   data/Structured/test/
   ```

   - Each class should have its own folder with images:
     ```
     data/Structured/train/walking/
     data/Structured/train/running/
     data/Structured/train/jumping/
     ```

2. The dataloader automatically detects class names from subfolders.

---

## Usage

### Training
Run training with LoRA adapters:
```bash
python train.py --model clip --batch_size 32 --epochs 20 --lr 1e-4 --num_classes 10
```

Arguments:
- `--model` : one of `clip`, `a_clip`, `sglip`
- `--batch_size` : default 32
- `--epochs` : number of training epochs
- `--lr` : learning rate
- `--num_classes` : number of dataset classes
- `--val_split` : fraction of training set for validation (default 0.2)

Outputs:
- Best checkpoint → `outputs/TRAIN##/{model}_best_checkpoint.pt`
- Metrics log → `outputs/TRAIN##/metrics.csv`
- Plots → `outputs/TRAIN##/*.png`
- Aggregated log → `outputs/all_trains_summary.csv`

---

### Evaluation
Evaluate a trained model:
```bash
python eval.py --model clip --batch_size 32 --num_classes 10
```

By default, this loads the **latest checkpoint** from `outputs/TRAIN##/`.

Outputs:
- Results summary → `outputs/EVAL##/results.csv`
- Per-class metrics → `outputs/EVAL##/{model}_test_per_class_metrics.csv`
- Confusion matrix + plots → `outputs/EVAL##/*.png`
- Aggregated log → `outputs/all_evals_summary.csv`

---

## OS-Specific Notes

### Windows
- Use `python` instead of `python3`
- Long paths may cause issues; consider enabling [long path support](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation)

### Linux / MacOS
- Use `python3` and `pip3`
- Ensure you have CUDA toolkit installed (`sudo apt install nvidia-cuda-toolkit` on Ubuntu)

---

## Monitoring
Launch TensorBoard to monitor training/eval:
```bash
tensorboard --logdir outputs
```

---

## Troubleshooting
- **OOM errors**: Lower `--batch_size`
- **CUDA not found**: Verify driver + CUDA version match
- **Slow training**: LoRA is enabled by default (`use_lora=True`), ensuring efficient GPU usage

---

## License
MIT — feel free to use and modify for research or development.
