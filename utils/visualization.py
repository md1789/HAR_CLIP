import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Confusion matrix saved at {save_path}")
    plt.close()


def generate_classification_report(y_true, y_pred, class_names, save_path=None):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:\n")
    print(report)
    if save_path:
        with open(save_path, "w") as f:
            f.write(report)
        print(f"Classification report saved at {save_path}")


def plot_loss_curve(train_losses, val_losses=None, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses,
             marker="o", label="Training Loss")
    if val_losses is not None:
        plt.plot(range(1, len(val_losses) + 1), val_losses,
                 marker="s", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Loss curve saved at {save_path}")
    plt.close()


def plot_class_metrics(y_true, y_pred, class_names, save_path=None):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names))
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x=class_names, y=class_acc, ax=ax[0], color="skyblue")
    ax[0].set_title("Per-Class Accuracy")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_ylim(0, 1)
    ax[0].set_xticklabels(class_names, rotation=45, ha="right")

    sns.barplot(x=class_names, y=f1, ax=ax[1], color="salmon")
    ax[1].set_title("Per-Class F1 Score")
    ax[1].set_ylabel("F1 Score")
    ax[1].set_ylim(0, 1)
    ax[1].set_xticklabels(class_names, rotation=45, ha="right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Per-class metrics plot saved at {save_path}")
    plt.close()


def compute_macro_micro(y_true, y_pred):
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro"
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    return {
        "micro": {"precision": p_micro, "recall": r_micro, "f1": f_micro},
        "macro": {"precision": p_macro, "recall": r_macro, "f1": f_macro},
    }

def log_per_class_metrics_to_tensorboard(writer, y_true, y_pred, class_names, split, step):
    """
    Logs per-class Precision, Recall, F1 to TensorBoard as bar plots.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names))
    )

    metrics = {"Precision": precision, "Recall": recall, "F1": f1}

    for metric_name, values in metrics.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=class_names, y=values, ax=ax, color="skyblue")
        ax.set_ylim(0, 1)
        ax.set_title(f"{split} Per-Class {metric_name}")
        ax.set_ylabel(metric_name)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        writer.add_figure(f"{split}/Per_Class_{metric_name}", fig, step)
        plt.close(fig)

def log_pr_curves_to_tensorboard(writer, y_true, y_pred, class_names, split, step):
    """
    Logs per-class Precision-Recall curves to TensorBoard.
    """
    num_classes = len(class_names)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for i, class_name in enumerate(class_names):
        # One-vs-rest ground truth
        true_binary = (y_true == i).astype(int)
        pred_binary = (y_pred == i).astype(int)

        if true_binary.sum() == 0:
            continue  # skip if class not present

        writer.add_pr_curve(f"{split}/PR_{class_name}", true_binary, pred_binary, step)

def log_roc_curves_to_tensorboard(writer, y_true, y_pred, class_names, split, step):
    """
    Logs per-class ROC curves and AUC to TensorBoard.
    """
    num_classes = len(class_names)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for i, class_name in enumerate(class_names):
        # One-vs-rest ground truth
        true_binary = (y_true == i).astype(int)
        pred_binary = (y_pred == i).astype(int)

        if true_binary.sum() == 0:
            continue  # skip if class not present

        fpr, tpr, _ = roc_curve(true_binary, pred_binary)
        roc_auc = auc(fpr, tpr)

        # Log AUC as scalar
        writer.add_scalar(f"{split}/ROC_AUC_{class_name}", roc_auc, step)

        # Log ROC curve as figure
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{split} ROC Curve - {class_name}")
        ax.legend(loc="lower right")
        writer.add_figure(f"{split}/ROC_{class_name}", fig, step)
        plt.close(fig)

def plot_accuracy_f1_curves(train_acc, val_acc, train_f1, val_f1, save_path=None):
    """
    Plot accuracy and F1 score curves for train and validation.
    """
    epochs = range(1, len(train_acc) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy curve
    ax[0].plot(epochs, train_acc, marker="o", label="Train Accuracy")
    ax[0].plot(epochs, val_acc, marker="s", label="Val Accuracy")
    ax[0].set_title("Accuracy Curve")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid(True)

    # F1 curve
    ax[1].plot(epochs, train_f1, marker="o", label="Train F1")
    ax[1].plot(epochs, val_f1, marker="s", label="Val F1")
    ax[1].set_title("F1 Score Curve")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("F1 Score")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Accuracy/F1 curves saved at {save_path}")
    plt.close()
