import tensorflow as tf
from tensorflow.keras import layers
from src.utils import EPOCHS_STAGE2


def fine_tune_model(
    model,
    base_model,
    train_ds,
    val_ds,
    epochs=EPOCHS_STAGE2
):
    """
    Stage 2 Fine-Tuning:
    Unfreezes the top 30 layers of EfficientNetB0 while keeping BatchNormalization
    layers frozen, and re-compiles with a reduced learning rate (1e-5).
    """
    print("\n" + "=" * 50)
    print("STAGE 2: FINE-TUNING EFFICIENTNETB0")
    print("=" * 50)

    base_model.trainable = True

    # Freeze all but the final 30 layers.
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Keep BatchNormalization layers frozen.
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )

    return history_stage2
