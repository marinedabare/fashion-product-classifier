import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

from config import Config
from src.dataset import prepare_data, create_tf_datasets, DISPLAY_NAMES
from src.models.scratch import ScratchClassifier
from src.models.pretrained import PretrainedClassifier


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def evaluate_model(model, dataset, model_name, config: Config):
    print(f"\n{'=' * 60}")
    print(f"EVALUATING: {model_name}")
    print(f"{'=' * 60}")

    loss, accuracy = model.evaluate(dataset, verbose=0)
    print(f"  Loss:     {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")

    all_preds = []
    all_labels = []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\n  Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=DISPLAY_NAMES)
    print(report)

    plot_confusion_matrix(
        all_labels,
        all_preds,
        DISPLAY_NAMES,
        f"Confusion Matrix — {model_name}",
        config.output_dir / f"cm_{model_name.lower().replace(' ', '_')}.png",
    )

    return {"loss": loss, "accuracy": accuracy}


def plot_comparison(results, config: Config):
    models = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in models]
    losses = [results[m]["loss"] for m in models]
    colors = ["#F44336", "#2196F3", "#4CAF50"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars1 = ax1.bar(models, accuracies, color=colors)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.1%}",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy")

    bars2 = ax2.bar(models, losses, color=colors)
    for bar, loss in zip(bars2, losses):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{loss:.4f}",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss (lower = more confident)")

    fig.suptitle("Three-Way Model Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_path = config.output_dir / "model_comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved comparison chart: {save_path}")


def main():
    config = Config()

    images, labels = prepare_data(config, samples_per_category=1000)
    (_, val_scratch), (_, val_pretrained) = create_tf_datasets(images, labels, config)

    model_files = {
        "From Scratch": ("scratch_final.keras", val_scratch),
        "Pretrained (Frozen)": ("pretrained_frozen.keras", val_pretrained),
        "Pretrained (Fine-tuned)": ("pretrained_finetuned.keras", val_pretrained),
    }

    results = {}
    for name, (filename, val_ds) in model_files.items():
        path = config.output_dir / filename
        if not path.exists():
            print(f"Skipping {name} — {filename} not found")
            continue
        model = tf.keras.models.load_model(str(path))
        results[name] = evaluate_model(model, val_ds, name, config)

    if len(results) > 1:
        plot_comparison(results, config)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for name, r in results.items():
        print(f"  {name:30s}  accuracy={r['accuracy']:.4f}  loss={r['loss']:.4f}")


if __name__ == "__main__":
    main()
