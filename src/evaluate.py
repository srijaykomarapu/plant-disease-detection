import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from src.utils import (
    OUTPUT_DIR,
    load_class_names
)


def evaluate_model(model, test_ds, class_names, output_dir=OUTPUT_DIR):
    """
    Evaluate the trained model on test dataset.
    Computes accuracy, precision, recall, f1 score, classification report,
    and generates the confusion matrix heatmap.
    """
    print("\n" + "=" * 50)
    print("TEST DATASET EVALUATION")
    print("=" * 50)

    num_classes = len(class_names)
    y_true_list = []
    y_pred_list = []

    for images, labels in test_ds:
        probabilities = model.predict(
            images,
            verbose=0
        )

        y_true_list.extend(
            np.argmax(labels.numpy(), axis=1)
        )

        y_pred_list.extend(
            np.argmax(probabilities, axis=1)
        )

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(f"\nAccuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        target_names=class_names,
        zero_division=0
    )

    print("\nClassification Report:\n")
    print(report)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes)
    )

    plt.figure(figsize=(16, 12))

    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("EfficientNetB0 Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(
        cm_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report_path": report_path,
        "confusion_matrix_path": cm_path
    }


if __name__ == "__main__":
    from src.data_preprocessing import load_datasets

    model_path = os.path.join(OUTPUT_DIR, "efficientnetb0_plant_disease.keras")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first.")

    model = tf.keras.models.load_model(model_path)
    class_names = load_class_names(OUTPUT_DIR)
    _, _, test_ds, _ = load_datasets()

    evaluate_model(model, test_ds, class_names, OUTPUT_DIR)
