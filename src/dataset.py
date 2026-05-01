import tensorflow as tf
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from config import Config

DISPLAY_NAMES = ["Shoes", "Watches", "Sunglasses", "Jeans"]

ARTICLE_TO_LABEL = {
    "Casual Shoes": 0,
    "Watches": 1,
    "Sunglasses": 2,
    "Jeans": 3,
}


def load_hf_dataset(samples_per_category=500):
    print("Loading dataset from Hugging Face (ashraq/fashion-product-images-small)...")
    ds = load_dataset("ashraq/fashion-product-images-small", split="train")

    images = []
    labels = []
    counts = {i: 0 for i in range(len(ARTICLE_TO_LABEL))}

    for item in ds:
        article = item.get("articleType", "")
        pil_img = item.get("image")

        if pil_img is None or article not in ARTICLE_TO_LABEL:
            continue

        label = ARTICLE_TO_LABEL[article]

        if counts[label] >= samples_per_category:
            if all(c >= samples_per_category for c in counts.values()):
                break
            continue

        images.append(pil_img)
        labels.append(label)
        counts[label] += 1

    for label_idx, name in enumerate(DISPLAY_NAMES):
        print(f"  {name}: {counts[label_idx]} samples")

    return images, labels


def prepare_data(config: Config, samples_per_category=500):
    pil_images, labels = load_hf_dataset(samples_per_category=samples_per_category)

    print(f"Resizing {len(pil_images)} images to {config.image_size}...")
    np_images = []
    valid_labels = []

    for i, pil_img in enumerate(pil_images):
        try:
            img = pil_img.convert("RGB").resize(
                (config.image_size[1], config.image_size[0])
            )
            np_images.append(np.array(img, dtype=np.float32))
            valid_labels.append(labels[i])
        except Exception:
            continue

    images = np.array(np_images, dtype=np.float32)
    labels = np.array(valid_labels, dtype=np.int32)

    print(f"Successfully loaded {len(images)} images.")
    return images, labels


augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
])


def augment_fn(image, label):
    return augment(image, training=True), label


def create_tf_datasets(images, labels, config: Config):
    img_train, img_val, y_train, y_val = train_test_split(
        images,
        labels,
        test_size=config.validation_split,
        random_state=config.seed,
        stratify=labels,
    )

    train_scratch = (
        tf.data.Dataset.from_tensor_slices((img_train / 255.0, y_train))
        .shuffle(1024)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_scratch = (
        tf.data.Dataset.from_tensor_slices((img_val / 255.0, y_val))
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    train_pretrained = (
        tf.data.Dataset.from_tensor_slices((
            tf.keras.applications.efficientnet.preprocess_input(img_train),
            y_train,
        ))
        .shuffle(1024)
        .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_pretrained = (
        tf.data.Dataset.from_tensor_slices((
            tf.keras.applications.efficientnet.preprocess_input(img_val),
            y_val,
        ))
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return (train_scratch, val_scratch), (train_pretrained, val_pretrained)
