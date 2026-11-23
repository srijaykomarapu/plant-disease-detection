import os
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ==========================================
# DEFAULT PATHS AND CONFIGURATION CONSTANTS
# ==========================================

DATASET_DIR = os.environ.get("DATASET_DIR", "/mnt/c/Users/chait/Downloads/PlantVillage")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.expanduser("~/PlantVillage_Results"))

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 10
SEED = 42


def set_seeds(seed=SEED):
    """Set random seed for reproducibility across tf, np, and random."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_gpu():
    """Enable GPU memory growth before GPU initialization."""
    print("TensorFlow version:", tf.__version__)
    print("Available GPUs:", tf.config.list_physical_devices("GPU"))
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print("GPU memory configuration:", error)


def verify_dataset_dirs(train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR):
    """Verify that local dataset folders exist."""
    print("Train:", train_dir)
    print("Validation:", val_dir)
    print("Test:", test_dir)
    print("Output:", OUTPUT_DIR)

    for name, path in [
        ("Train", train_dir),
        ("Validation", val_dir),
        ("Test", test_dir)
    ]:
        print(f"{name} folder exists:", os.path.isdir(path))

    for name, path in [
        ("Train", train_dir),
        ("Validation", val_dir),
        ("Test", test_dir)
    ]:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"{name} folder not found:\n{path}\n"
                "Please check DATASET_DIR and folder names."
            )
    print("\nAll dataset folders verified.")


def save_class_names(class_names, output_dir=OUTPUT_DIR):
    """Save dataset class names to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "efficientnetb0_classes.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)
    return json_path


def load_class_names(output_dir=OUTPUT_DIR):
    """Load dataset class names from a JSON file."""
    json_path = os.path.join(output_dir, "efficientnetb0_classes.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Class names file not found at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_history(history_stage1, history_stage2, output_dir=OUTPUT_DIR):
    """Plot and save training and validation accuracy and loss curves."""
    train_accuracy = (
        history_stage1.history["accuracy"]
        + history_stage2.history["accuracy"]
    )
    val_accuracy = (
        history_stage1.history["val_accuracy"]
        + history_stage2.history["val_accuracy"]
    )
    train_loss = (
        history_stage1.history["loss"]
        + history_stage2.history["loss"]
    )
    val_loss = (
        history_stage1.history["val_loss"]
        + history_stage2.history["val_loss"]
    )

    epochs_range = range(1, len(train_accuracy) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_accuracy, label="Training Accuracy")
    plt.plot(epochs_range, val_accuracy, label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    history_plot_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(
        history_plot_path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()
    return history_plot_path
