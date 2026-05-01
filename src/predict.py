import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

from config import Config
from src.dataset import DISPLAY_NAMES
from src.models.scratch import ScratchClassifier
from src.models.pretrained import PretrainedClassifier
from src.models.description_generator import DescriptionGenerator


def load_image(image_path, image_size):
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size)
    return tf.cast(img, tf.float32)


def classify(model, image, use_efficientnet=False):
    if use_efficientnet:
        img = tf.keras.applications.efficientnet.preprocess_input(image)
    else:
        img = image / 255.0
    img_batch = tf.expand_dims(img, 0)
    predictions = model.predict(img_batch, verbose=0)
    return predictions[0]


def display_results(predictions, model_name):
    print(f"\n--- {model_name} ---")
    sorted_indices = np.argsort(predictions)[::-1]
    for idx in sorted_indices:
        bar = "█" * int(predictions[idx] * 30)
        print(f"  {DISPLAY_NAMES[idx]:15s} {predictions[idx]:6.1%} {bar}")
    top = sorted_indices[0]
    print(f"  → Prediction: {DISPLAY_NAMES[top]} ({predictions[top]:.1%})")
    return DISPLAY_NAMES[top]


def main():
    parser = argparse.ArgumentParser(description="Classify a product image")
    parser.add_argument("--image", type=str, required=True, help="Path to product image")
    parser.add_argument(
        "--model",
        choices=["scratch", "frozen", "finetuned", "all"],
        default="all",
    )
    args = parser.parse_args()

    config = Config()
    image = load_image(args.image, config.image_size)

    print(f"Image: {args.image}")

    model_map = {
        "scratch": ("From Scratch", "scratch_final.keras"),
        "frozen": ("Pretrained (Frozen)", "pretrained_frozen.keras"),
        "finetuned": ("Pretrained (Fine-tuned)", "pretrained_finetuned.keras"),
    }

    models_to_run = model_map if args.model == "all" else {args.model: model_map[args.model]}

    for key, (name, filename) in models_to_run.items():
        path = config.output_dir / filename
        if not path.exists():
            print(f"\nSkipping {name} — {filename} not found. Train first.")
            continue
        model = tf.keras.models.load_model(str(path))
        use_effnet = key in ("frozen", "finetuned")
        preds = classify(model, image, use_efficientnet=use_effnet)
        display_results(preds, name)

    print(f"\n--- Generated Description ---")
    generator = DescriptionGenerator()
    pil_image = Image.open(args.image).convert("RGB")
    description = generator.generate(pil_image)
    print(f"  {description}")


if __name__ == "__main__":
    main()
