from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np


class DescriptionGenerator:
    """
    Uses BLIP (Bootstrapped Language-Image Pretraining) to generate
    a text description from a product image. This model is pretrained
    and used as-is — no fine-tuning needed.
    """

    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

    def generate(self, image, max_length=50):
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        inputs = self.processor(
            images=image,
            text="a product photo of",
            return_tensors="pt",
        )

        output_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
        )

        description = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return description
