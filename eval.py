import argparse
import os
import glob
import csv
import torch
import pandas as pd
from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import get_model
from utils.common import get_dataloader, compute_metrics
from utils.visualization import (
    plot_confusion_matrix,
    generate_classification_report,
    plot_class_metrics,
    compute_macro_micro,
    log_per_class_metrics_to_tensorboard,
    log_pr_curves_to_tensorboard,
    log_roc_curves_to_tensorboard
)
from sklearn.metrics import precision_recall_fscore_support


def get_latest_checkpoint(base_dir="outputs", prefix="TRAIN"):
    if not os.path.exists(base_dir):
        return None
    run_folders = [f for f in os.listdir(base_dir) if f.startswith(prefix)]
    if not run_folders:
        return None
    run_folders.sort()
    latest_run = run_folders[-1]
    run_path = os.path.join(base_dir, latest_run)

    best_ckpts = [f for f in os.listdir(run_path) if f.endswith("_best_checkpoint.pt")]
    if best_ckpts:
        return os.path.join(run_path, best_ckpts[-1])

    ckpts = [f for f in os.listdir(run_path) if f.endswith("_checkpoint.pt")]
    if not ckpts:
        return None
    ckpts.sort()
    return os.path.join(run_path, ckpts[-1])


def get_next_run_folder(base_dir="outputs", prefix="EVAL"):
    os.makedirs(base_dir, exist_ok=True)
    run_id = 1
    while True:
        run_folder = os.path.join(base_dir, f"{prefix}{run_id:02d}")
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
            return run_folder, f"{prefix}{run_id:02d}"
        run_id += 1


def run_eval_split(model, loader, class_names, split: str,
                   eval_dir: str, eval_id: str, train_run: str,
                   model_name: str, device: str, csv_writer=None,
                   tb_writer=None, step=0):

    loss_fn = CrossEntropyLoss()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_preds, all_labels)
    avg_scores = compute_macro_micro(all_labels, all_preds)

    # Save summary row
    if csv_writer:
        csv_writer.writerow([
            eval_id, train_run, split,
            avg_loss, metrics["accuracy"],
            avg_scores["micro"]["precision"], avg_scores["micro"]["recall"], avg_scores["micro"]["f1"],
            avg_scores["macro"]["precision"], avg_scores["macro"]["recall"], avg_scores["macro"]["f1"]
        ])

    # Save per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(class_names))
    )
    per_class_csv = os.path.join(eval_dir, f"{model_name}_{split}_per_class_metrics.csv")
    with open(per_class_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for cls, p, r, f_, s in zip(class_names, precision, recall, f1, support):
            writer.writerow([cls, p, r, f_, s])

    print(f"\n[{model_name}] {split.upper()} RESULTS")
    print(f"Loss: {avg_loss:.4f} - Accuracy: {metrics['accuracy']:.4f}")
    print(f"Micro Avg - Precision: {avg_scores['micro']['precision']:.4f}, "
          f"Recall: {avg_scores['micro']['recall']:.4f}, "
          f"F1: {avg_scores['micro']['f1']:.4f}")
    print(f"Macro Avg - Precision: {avg_scores['macro']['precision']:.4f}, "
          f"Recall: {avg_scores['macro']['recall']:.4f}, "
          f"F1: {avg_scores['macro']['f1']:.4f}")

    # Save plots/reports
    cm_path = os.path.join(eval_dir, f"{model_name}_{split}_confusion_matrix.png")
    report_path = os.path.join(eval_dir, f"{model_name}_{split}_classification_report.txt")
    class_metrics_path = os.path.join(eval_dir, f"{model_name}_{split}_class_metrics.png")

    plot_confusion_matrix(all_labels, all_preds, class_names, save_path=cm_path)
    generate_classification_report(all_labels, all_preds, class_names, save_path=report_path)
    plot_class_metrics(all_labels, all_preds, class_names, save_path=class_metrics_path)

    # TensorBoard logging
    if tb_writer:
        tb_writer.add_scalar(f"{split}/Loss", avg_loss, step)
        tb_writer.add_scalar(f"{split}/Accuracy", metrics["accuracy"], step)
        tb_writer.add_scalar(f"{split}/Micro_F1", avg_scores["micro"]["f1"], step)
        tb_writer.add_scalar(f"{split}/Macro_F1", avg_scores["macro"]["f1"], step)

        log_per_class_metrics_to_tensorboard(tb_writer, all_labels, all_preds, class_names, split, step)
        log_pr_curves_to_tensorboard(tb_writer, all_labels, all_preds, class_names, split, step)
        log_roc_curves_to_tensorboard(tb_writer, all_labels, all_preds, class_names, split, step)

    return metrics


def evaluate(model_name: str, batch_size: int, num_classes: int,
             checkpoint_path: str = None, val_split: float = 0.2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load base frozen model
    model = get_model(model_name, num_classes=num_classes, use_lora=True).to(device)
    preprocess = getattr(model, "preprocess", None)

    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint()
        if checkpoint_path:
            print(f"Auto-detected checkpoint: {checkpoint_path}")
            train_run = os.path.basename(os.path.dirname(checkpoint_path))
        else:
            raise FileNotFoundError("No checkpoint found in outputs/ — please train first.")
    else:
        train_run = os.path.basename(os.path.dirname(checkpoint_path))

    # Load LoRA adapter weights
    lora_state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(lora_state, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")

    model.eval()
    eval_dir, eval_id = get_next_run_folder(base_dir="outputs", prefix="EVAL")
    tb_writer = SummaryWriter(log_dir=eval_dir)

    # CSV setup
    csv_path = os.path.join(eval_dir, "results.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "eval_run", "train_run", "split", "loss", "accuracy",
        "micro_precision", "micro_recall", "micro_f1",
        "macro_precision", "macro_recall", "macro_f1"
    ])

    # Train/Val split
    full_loader, full_dataset = get_dataloader(
        root="data/Structured",
        split="train",
        batch_size=batch_size,
        preprocess=preprocess,
        shuffle=False
    )
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    class_names = full_dataset.classes

    # Test set
    test_loader, test_dataset = get_dataloader(
        root="data/Structured",
        split="test",
        batch_size=batch_size,
        preprocess=preprocess,
        shuffle=False
    )

    results = {}
    results["train"] = run_eval_split(model, train_loader, class_names, "train",
                                      eval_dir, eval_id, train_run, model_name, device, csv_writer, tb_writer, step=1)
    results["val"] = run_eval_split(model, val_loader, class_names, "val",
                                    eval_dir, eval_id, train_run, model_name, device, csv_writer, tb_writer, step=1)
    results["test"] = run_eval_split(model, test_loader, test_dataset.classes, "test",
                                     eval_dir, eval_id, train_run, model_name, device, csv_writer, tb_writer, step=1)

    csv_file.close()
    tb_writer.close()

    # --- Merge step: aggregate all eval results ---
    all_csvs = glob.glob("outputs/EVAL*/results.csv")
    dfs = []
    for path in sorted(all_csvs):
        df = pd.read_csv(path)
        dfs.append(df)

    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged_path = os.path.join("outputs", "all_evals_summary.csv")
        merged.to_csv(merged_path, index=False)
        print(f"All evaluations merged into {merged_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["clip", "a_clip", "sglip"],
                        help="Which model to evaluate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (.pt). If not provided, auto-detects best/latest.")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Proportion of training data to use for validation")
    args = parser.parse_args()

    evaluate(args.model, args.batch_size, args.num_classes, args.checkpoint, args.val_split)
