import os
import glob
import numpy as np
import tensorflow as tf
from src.utils import (
    IMAGE_SIZE,
    TEST_DIR,
    OUTPUT_DIR,
    load_class_names
)


def predict_single_leaf(
    image_path,
    model_obj,
    labels_list,
    threshold=80.0
):
    """
    Predict plant disease for a single leaf image.
    Applies confidence thresholding (default 80.0%) and healthy/diseased categorization.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    raw_img = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    img_array = tf.keras.utils.img_to_array(raw_img)
    img_batch = np.expand_dims(img_array, axis=0)

    probabilities = model_obj.predict(
        img_batch,
        verbose=0
    )[0]

    top_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[top_idx]) * 100
    predicted_category = labels_list[top_idx]

    if confidence < threshold:
        return {
            "status": "Low Confidence / Uncertain",
            "predicted_category": predicted_category,
            "confidence": f"{confidence:.2f}%"
        }

    health_status = (
        "Healthy"
        if "healthy" in predicted_category.lower()
        else "Diseased"
    )

    return {
        "predicted_disease": predicted_category,
        "health_status": health_status,
        "confidence": f"{confidence:.2f}%"
    }


def predict_sample(model=None, class_names=None, test_dir=TEST_DIR):
    """Find and predict the first available sample image from the test directory."""
    if model is None:
        model_path = os.path.join(OUTPUT_DIR, "efficientnetb0_plant_disease.keras")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first.")
        model = tf.keras.models.load_model(model_path)

    if class_names is None:
        class_names = load_class_names(OUTPUT_DIR)

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    sample_images = []

    for extension in extensions:
        sample_images.extend(
            glob.glob(
                os.path.join(test_dir, "*", extension)
            )
        )

    sample_images.sort()

    if sample_images:
        test_file = sample_images[0]

        result = predict_single_leaf(
            test_file,
            model,
            class_names,
            threshold=80.0
        )

        print("\nSAMPLE LEAF PREDICTION")
        print("Image:", test_file)

        for key, value in result.items():
            print(f"{key}: {value}")
        return result
    else:
        print(
            "\nNo test images found directly inside class folders. "
            "Check the test dataset structure."
        )
        return None


if __name__ == "__main__":
    predict_sample()
