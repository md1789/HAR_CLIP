import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from models import get_model
from utils.common import get_dataloader, compute_metrics
from utils.visualization import (
    plot_loss_curve, plot_accuracy_f1_curves,
    log_per_class_metrics_to_tensorboard,
    log_pr_curves_to_tensorboard,
    log_roc_curves_to_tensorboard
)
import csv
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np


def get_next_run_folder(base_dir="outputs", prefix="TRAIN"):
    os.makedirs(base_dir, exist_ok=True)
    run_id = 1
    while True:
        run_folder = os.path.join(base_dir, f"{prefix}{run_id:02d}")
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
            return run_folder
        run_id += 1


def train(model_name: str, batch_size: int, epochs: int, lr: float,
          num_classes: int, val_split: float = 0.2, patience: int = 5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_name, num_classes=num_classes, use_lora=True).to(device)

    # --- Sanity check for LoRA ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[LoRA Check] Total params: {total_params:,}")
    print(f"[LoRA Check] Trainable params: {trainable_params:,}")
    print(f"[LoRA Check] Percentage trainable: {100 * trainable_params / total_params:.2f}%\n")

    preprocess = getattr(model, "preprocess", None)

    # Freeze everything except LoRA
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Load full dataset
    full_loader, full_dataset = get_dataloader(
        root="data/Structured",
        split="train",
        batch_size=batch_size,
        preprocess=preprocess,
        shuffle=True
    )

    print("Checking dataset...")
    print("Train dataset size:", len(train_dataset))
    print("Classes:", full_dataset.classes)
    print("Num classes arg:", num_classes)

    all_labels = [label for _, label in train_dataset]
    print("Label min:", min(all_labels), "Label max:", max(all_labels))

    # Train/Val split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Optimizer over only LoRA params
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = CrossEntropyLoss()

    # Output paths
    run_folder = get_next_run_folder(base_dir="outputs", prefix="TRAIN")
    checkpoint_path = os.path.join(run_folder, f"{model_name}_best_checkpoint.pt")
    loss_curve_path = os.path.join(run_folder, f"{model_name}_loss_curve.png")
    acc_f1_curve_path = os.path.join(run_folder, f"{model_name}_accuracy_f1_curves.png")

    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir=run_folder)

    # Output CSV path (create with header)
    csv_path = os.path.join(run_folder, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "epoch",
            "train_loss", "val_loss",
            "train_acc", "val_acc",
            "train_f1", "val_f1"
        ])

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_train_loss = 0
        all_train_preds, all_train_labels = [], []

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_train_preds.append(preds.cpu())
            all_train_labels.append(labels.cpu())

            # live progress bar updates
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        all_train_preds = torch.cat(all_train_preds)
        all_train_labels = torch.cat(all_train_labels)
        train_metrics = compute_metrics(all_train_preds, all_train_labels)

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = loss_fn(logits, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_val_preds.append(preds.cpu())
                all_val_labels.append(labels.cpu())

        avg_val_loss = total_val_loss / len(val_loader)
        all_val_preds = torch.cat(all_val_preds)
        all_val_labels = torch.cat(all_val_labels)
        val_metrics = compute_metrics(all_val_preds, all_val_labels)

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_metrics["accuracy"])
        val_accs.append(val_metrics["accuracy"])
        train_f1s.append(train_metrics["f1"])
        val_f1s.append(val_metrics["f1"])

        # --- Save metrics row to CSV ---
        with open(csv_path, "a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                epoch+1,
                avg_train_loss, avg_val_loss,
                train_metrics["accuracy"], val_metrics["accuracy"],
                train_metrics["f1"], val_metrics["f1"]
            ])

        print(f"[{model_name}] Epoch {epoch+1}/{epochs} "
              f"- Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} "
              f"- Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f} "
              f"- Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")

        # --- TensorBoard logging ---
        tb_writer.add_scalar("Loss/Train", avg_train_loss, epoch+1)
        tb_writer.add_scalar("Loss/Val", avg_val_loss, epoch+1)
        tb_writer.add_scalar("Accuracy/Train", train_metrics["accuracy"], epoch+1)
        tb_writer.add_scalar("Accuracy/Val", val_metrics["accuracy"], epoch+1)
        tb_writer.add_scalar("F1/Train", train_metrics["f1"], epoch+1)
        tb_writer.add_scalar("F1/Val", val_metrics["f1"], epoch+1)

        # Log per-class + curves for validation set
        log_per_class_metrics_to_tensorboard(tb_writer, all_val_labels, all_val_preds,
                                             full_dataset.classes, "val", epoch+1)
        log_pr_curves_to_tensorboard(tb_writer, all_val_labels, all_val_preds,
                                     full_dataset.classes, "val", epoch+1)
        log_roc_curves_to_tensorboard(tb_writer, all_val_labels, all_val_preds,
                                      full_dataset.classes, "val", epoch+1)

        # --- Save best checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save only LoRA params
            torch.save(
                {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k},
                checkpoint_path
            )
            print(f"New best model saved at {checkpoint_path} (val_loss={best_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Update plots
        plot_loss_curve(train_losses, val_losses, save_path=loss_curve_path)
        plot_accuracy_f1_curves(train_accs, val_accs, train_f1s, val_f1s, save_path=acc_f1_curve_path)

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best val_loss={best_val_loss:.4f}")
            break

    # --- Merge step: aggregate all training metrics into one CSV ---
    all_csvs = glob.glob("outputs/TRAIN*/metrics.csv")
    dfs = []
    for path in sorted(all_csvs):
        train_id = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path)
        df.insert(0, "train_run", train_id)
        dfs.append(df)

    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged_path = os.path.join("outputs", "all_trains_summary.csv")
        merged.to_csv(merged_path, index=False)
        print(f"All training runs merged into {merged_path}")

    tb_writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["clip", "a_clip", "sglip"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    train(args.model, args.batch_size, args.epochs, args.lr, args.num_classes, args.val_split)

