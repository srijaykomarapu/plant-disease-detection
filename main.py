"""
Main pipeline runner for Plant Disease Detection with EfficientNetB0.
Executes the full workflow: training, evaluation, and sample prediction.
"""

from src.train import train
from src.evaluate import evaluate_model
from src.predict import predict_sample
from src.utils import OUTPUT_DIR


def main():
    # 1. Train model (Stage 1 Feature Extraction & Stage 2 Fine-Tuning)
    model, class_names, test_ds = train()

    # 2. Evaluate model on test dataset
    evaluate_model(model, test_ds, class_names, OUTPUT_DIR)

    # 3. Predict sample leaf image
    predict_sample(model, class_names)

    # 4. Completion
    print("\n" + "=" * 50)
    print("ALL OPERATIONS COMPLETED")
    print("Results saved locally at:")
    print(OUTPUT_DIR)
    print("=" * 50)


if __name__ == "__main__":
    main()
