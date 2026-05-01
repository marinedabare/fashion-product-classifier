import inspect
_orig_getattr_static = inspect.getattr_static
def _safe_getattr_static(obj, attr, *args):
    try:
        return _orig_getattr_static(obj, attr, *args)
    except TypeError:
        return args[0] if args else None
inspect.getattr_static = _safe_getattr_static

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

from config import Config
from src.dataset import DISPLAY_NAMES
from src.models.scratch import ScratchClassifier
from src.models.pretrained import PretrainedClassifier
from src.models.description_generator import DescriptionGenerator

config = Config()

models = {}
generator = None


def load_all():
    global models, generator

    model_files = {
        "From Scratch": "scratch_final.keras",
        "Pretrained (Frozen)": "pretrained_frozen.keras",
        "Pretrained (Fine-tuned)": "pretrained_finetuned.keras",
    }

    for name, filename in model_files.items():
        path = config.output_dir / filename
        if path.exists():
            models[name] = tf.keras.models.load_model(str(path))
            print(f"Loaded: {name}")

    generator = DescriptionGenerator()
    print("Loaded: BLIP description generator")


def preprocess_image(image, use_efficientnet=False):
    img = tf.image.resize(image, config.image_size)
    img = tf.cast(img, tf.float32)
    if use_efficientnet:
        img = tf.keras.applications.efficientnet.preprocess_input(img)
    else:
        img = img / 255.0
    return tf.expand_dims(img, 0)


def classify(image):
    if image is None:
        return {}, {}, {}, "Please upload an image."

    img_scratch = preprocess_image(image, use_efficientnet=False)
    img_pretrained = preprocess_image(image, use_efficientnet=True)

    results = {}
    for name, model in models.items():
        batch = img_scratch if name == "From Scratch" else img_pretrained
        preds = model.predict(batch, verbose=0)[0]
        results[name] = {DISPLAY_NAMES[i]: float(preds[i]) for i in range(len(DISPLAY_NAMES))}

    pil_image = Image.fromarray(image.astype(np.uint8))
    description = generator.generate(pil_image)

    return (
        results.get("From Scratch", {"No model": 1.0}),
        results.get("Pretrained (Frozen)", {"No model": 1.0}),
        results.get("Pretrained (Fine-tuned)", {"No model": 1.0}),
        description,
    )


def main():
    load_all()

    with gr.Blocks(title="Fashion Product Classifier") as demo:
        gr.Markdown(
            "# Fashion Product Classifier\n"
            "*Trained with 1000 Shoes, 1000 Watches, 1000 Sunglasses, 602 Jeans*\n\n"
            "Upload a product photo on a clean background "
            "(Shoes, Watches, Sunglasses, Jeans) "
            "to classify it and generate a description. "
            "Compare results across three models: from scratch, "
            "pretrained frozen, and pretrained fine-tuned."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="numpy",
                    label="Product Image",
                    sources=["upload"],
                    height=600,
                    buttons=[],
                )
                submit_btn = gr.Button("Classify", variant="primary", interactive=False)
            with gr.Column():
                label_scratch = gr.Label(num_top_classes=4, label="From Scratch")
                label_frozen = gr.Label(num_top_classes=4, label="Pretrained (Frozen)")
                label_finetuned = gr.Label(num_top_classes=4, label="Pretrained (Fine-tuned)")
                description_out = gr.Textbox(label="Generated Description")

        image_input.change(
            fn=lambda img: gr.update(interactive=img is not None),
            inputs=image_input,
            outputs=submit_btn,
        )

        submit_btn.click(
            fn=classify,
            inputs=image_input,
            outputs=[label_scratch, label_frozen, label_finetuned, description_out],
        )

    demo.launch(ssr_mode=False)


if __name__ == "__main__":
    main()
