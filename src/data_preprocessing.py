import os
import tensorflow as tf
from tensorflow.keras import layers, models
from src.utils import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED
)

# ==========================================
# DATA AUGMENTATION PIPELINE
# ==========================================

data_augmentation = models.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15)
    ],
    name="data_augmentation"
)


def load_datasets(
    train_dir=TRAIN_DIR,
    val_dir=VAL_DIR,
    test_dir=TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
):
    """Load train, validation, and test datasets with tf.keras.utils."""
    print("\nLoading training dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=seed
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print("\nLoading validation dataset...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        class_names=class_names,
        shuffle=False
    )

    print("\nLoading test dataset...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        class_names=class_names,
        shuffle=False
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    print("\nClass names:", class_names)
    print("Number of classes:", num_classes)

    return train_ds, val_ds, test_ds, class_names
