import tensorflow as tf
from tensorflow.keras import layers, models
from src.utils import IMAGE_SIZE
from src.data_preprocessing import data_augmentation


def build_model(num_classes, image_size=IMAGE_SIZE, augmentation=data_augmentation):
    """
    Build EfficientNetB0 model with custom classification head.
    Initializes Stage 1 configuration where base_model layers are frozen.
    """
    print("\nBuilding EfficientNetB0 model...")

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights="imagenet"
    )

    # Stage 1: freeze pretrained layers.
    base_model.trainable = False

    inputs = layers.Input(shape=(image_size[0], image_size[1], 3))

    x = augmentation(inputs) if augmentation is not None else inputs

    # EfficientNetB0 includes its own input rescaling.
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(inputs, outputs)
    model.summary()

    return model, base_model
