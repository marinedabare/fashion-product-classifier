---
title: Fashion Product Classifier
emoji: 🌖
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.10.0
app_file: app.py
python_version: "3.12"
pinned: false
license: mit
short_description: Fashion product classifier with model comparison
---

# Fashion Product Classifier

An image classification project comparing **training from scratch vs transfer learning** on a real-world dataset. Includes a BLIP-powered description generator. Built with TensorFlow and Gradio.

## What it does

Upload a product image → get back:
1. **Category prediction** (Shoes, Watches, Sunglasses, Jeans)
2. **Generated product description** (powered by BLIP vision-language model)

## Three-Way Model Comparison

This project trains and compares three classification approaches to demonstrate the impact of transfer learning:

| Model | Approach | What it proves |
|---|---|---|
| **From Scratch** | Custom CNN, trained from zero | Baseline — what raw data alone can learn |
| **Pretrained (Frozen)** | EfficientNetB0, backbone not updated | What ImageNet knowledge gives you for free |
| **Pretrained (Fine-tuned)** | EfficientNetB0, backbone partially retrained | Best of both — general knowledge adapted to our task |

### What each model handles

| Step | From Scratch | Pretrained (Frozen) | Pretrained (Fine-tuned) |
|---|---|---|---|
| **Classification** | Custom CNN (from zero) | EfficientNet (frozen) | EfficientNet (fine-tuned) |
| **Description** | BLIP (pretrained) | BLIP (pretrained) | BLIP (pretrained) |

The three-way comparison focuses on classification. Description generation uses BLIP across all models — training a text generator from scratch would require millions of samples and is out of scope.

## Architecture

```
┌─────────────────┐
│  Product Image   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌─────────────────────┐
│ CNN /  │ │ BLIP Vision-Language │
│ EffNet │ │ Model                │
└───┬────┘ └──────────┬──────────┘
    │                 │
    ▼                 ▼
┌────────┐ ┌─────────────────────┐
│Category│ │ Generated Product    │
│(1 of 4)│ │ Description          │
└────────┘ └─────────────────────┘
```

### From-Scratch CNN
- Conv2D (32 → 64 → 128 → 256) + MaxPooling + GlobalAveragePooling
- Dense (256 → 128) + BatchNorm + Dropout (0.5) → Softmax

### Pretrained (EfficientNetB0)
- EfficientNetB0 backbone (ImageNet weights)
- Same classification head as above
- Two-phase training: frozen backbone → fine-tune last 10 layers

### Description Generator (BLIP)
- Salesforce BLIP pretrained vision-language model
- Generates captions from images with beam search

## Dataset

- **Source:** [Fashion Product Images](https://huggingface.co/datasets/ashraq/fashion-product-images-small) (Hugging Face)
- **Categories:** Shoes, Watches, Sunglasses, Jeans
- **Size:** 1,000 images per category (Jeans capped at 602), ~3,600 total
- **Downloaded automatically** on first training run — no manual setup needed

## How to Run

### Try the live demo (no setup needed)

The app is deployed on Hugging Face Spaces — just upload an image:

**[Live Demo](https://huggingface.co/spaces/marinedabare/fashion-product-classifier)**

### Train on Google Colab (GPU recommended)

Open the training notebook in Colab for fast GPU training:

1. Open `notebooks/train_colab.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type)
3. Run all cells — downloads data, trains 3 models, evaluates, and pushes results

### Run locally

**Requires Python 3.10–3.13** (TensorFlow does not support Python 3.14+ yet).

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Train and evaluate:
```bash
python -m src.train        # ~30-60 min on CPU
python -m src.evaluate     # generates comparison charts and confusion matrices
```

Launch the web demo locally:
```bash
python app.py
```

Classify a single image:
```bash
python -m src.predict --image path/to/product.jpg --model all
```

Run tests:
```bash
pytest tests/
```

## Results

| Model | Accuracy | Loss |
|---|---|---|
| From Scratch | 98.9% | 0.0256 |
| Pretrained (Frozen) | 99.7% | 0.0208 |
| Pretrained (Fine-tuned) | 99.7% | 0.0166 |

All three models perform well on these 4 visually distinct categories. The pretrained models reach near-perfect accuracy in just a few epochs, while the from-scratch CNN needs 30 epochs to converge. Although frozen and fine-tuned share the same accuracy, the fine-tuned model achieves a lower loss, meaning it is more confident in its predictions.

The from-scratch model is slightly less accurate on Jeans, which makes sense — the dataset only contains 602 jeans images compared to 1,000 for the other categories. Less training data means the model has fewer examples to learn from, and this category is the first to suffer. The pretrained models are not affected because they already have a strong visual understanding from ImageNet.

Confusion matrices and comparison charts are saved to `outputs/`.

## Project Structure

```
├── config.py                          # Hyperparameters
├── app.py                             # Gradio web demo
├── requirements.txt                   # Dependencies
├── src/
│   ├── dataset.py                     # Data loading from Hugging Face
│   ├── train.py                       # Train all three models
│   ├── evaluate.py                    # Evaluate and compare
│   ├── predict.py                     # Single-image inference
│   └── models/
│       ├── scratch.py                 # Custom CNN (from zero)
│       ├── pretrained.py              # EfficientNetB0 classifier
│       └── description_generator.py   # BLIP caption generator
└── tests/
    └── test_model.py                  # Unit tests
```

## Key Takeaways

### Why transfer learning dominates with limited data

The from-scratch CNN must learn everything from zero — edges, shapes, textures, colors — using only ~3,600 images. It performs well on these 4 visually distinct categories (98.9%), but the features it learns are less robust and would struggle to generalize to new categories or more varied images compared to a pretrained model.

**Tackling overfitting in the from-scratch model:** Early experiments showed the from-scratch CNN was memorizing training images instead of learning general patterns (97% train accuracy but only 27% validation accuracy). Two changes fixed this:
- **Higher early stopping patience** (10 instead of 5): the model goes through a chaotic early phase where validation accuracy fluctuates wildly. With patience=5, training was killed before the model had time to stabilize. With patience=10, it had enough time to converge and reach 98.9%.
- **Gradual learning rate reduction** (ReduceLROnPlateau with patience=5): as training progressed, the learning rate was automatically lowered at the right moments, which smoothed out the training and helped the model generalize instead of memorize.

The pretrained models (EfficientNetB0) don't need any of this. They already know what edges, shapes, and textures look like, learned from ImageNet's 1.2 million images. They only need to learn which combination of features maps to "shoes" vs "watches" — a much simpler task that converges in just a few epochs.

### Why frozen ≈ fine-tuned at this scale

Fine-tuning unlocks the last 10 layers of EfficientNetB0 and retrains them with a very low learning rate (2e-5), allowing the model to adapt its high-level features to our specific domain. However, with under 5,000 images, there isn't enough data for this adaptation to improve on what ImageNet already provides. The fashion categories (shoes, watches, sunglasses, jeans) are visually distinct and well-represented in ImageNet, so the frozen features are already near-optimal.

Fine-tuning becomes valuable with **larger datasets (5,000–20,000+ images)** or **specialized domains far from ImageNet** (e.g. medical imaging, satellite photos), where the pretrained features need significant adaptation.

We deliberately kept this project small-scale for practical reasons: a larger dataset (20,000+ images) would require ~12 GB of RAM and several hours of training, making it impractical to run on CPU or free-tier Colab. Using a specialized domain (e.g. medical images) would have shown a bigger gap between frozen and fine-tuned, but wouldn't be representative of a typical image classification task — and the three-way comparison already tells a clear, honest story about when transfer learning helps and when fine-tuning isn't worth the extra cost.


## Tech Stack

- TensorFlow / Keras — classification models
- EfficientNetB0 — pretrained image backbone (ImageNet)
- BLIP (Hugging Face Transformers) — description generation
- Hugging Face Datasets — data loading
- Gradio — interactive web demo
- scikit-learn — evaluation metrics
