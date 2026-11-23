import os
import tensorflow as tf
from src.utils import (
    set_seeds,
    setup_gpu,
    verify_dataset_dirs,
    save_class_names,
    plot_training_history,
    OUTPUT_DIR,
    EPOCHS_STAGE1
)
from src.data_preprocessing import load_datasets
from src.model import build_model
from src.fine_tune import fine_tune_model


def train():
    """Execute complete two-stage training pipeline."""
    print("=" * 50)
    print("PLANT DISEASE DETECTION - EFFICIENTNETB0")
    print("=" * 50)

    set_seeds()
    setup_gpu()
    verify_dataset_dirs()

    # Load datasets
    train_ds, val_ds, test_ds, class_names = load_datasets()
    num_classes = len(class_names)

    # Save class mapping
    save_class_names(class_names, OUTPUT_DIR)

    # Build model (Stage 1: Frozen base model)
    model, base_model = build_model(num_classes)

    # Stage 1: Feature Extraction
    print("\n" + "=" * 50)
    print("STAGE 1: TRAINING CLASSIFICATION HEAD")
    print("=" * 50)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        verbose=1
    )

    # Save Stage 1 model
    stage1_path = os.path.join(OUTPUT_DIR, "efficientnetb0_stage1.keras")
    model.save(stage1_path)
    print(f"Stage 1 model saved: {stage1_path}")

    # Stage 2: Fine-Tuning
    history_stage2 = fine_tune_model(
        model=model,
        base_model=base_model,
        train_ds=train_ds,
        val_ds=val_ds
    )

    # Save final model
    final_model_path = os.path.join(OUTPUT_DIR, "efficientnetb0_plant_disease.keras")
    model.save(final_model_path)
    save_class_names(class_names, OUTPUT_DIR)

    print("\nFinal model saved:", final_model_path)
    print("Results directory:", OUTPUT_DIR)

    # Plot training history curves
    plot_training_history(history_stage1, history_stage2, OUTPUT_DIR)

    return model, class_names, test_ds


if __name__ == "__main__":
    train()
