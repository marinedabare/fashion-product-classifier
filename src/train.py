import tensorflow as tf
from pathlib import Path

from config import Config
from src.dataset import prepare_data, create_tf_datasets
from src.models.scratch import ScratchClassifier
from src.models.pretrained import PretrainedClassifier


def get_callbacks(model_name, output_dir):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / f"{model_name}_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(output_dir / "logs" / model_name),
        ),
    ]


def get_scratch_callbacks(output_dir):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "scratch_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(output_dir / "logs" / "scratch"),
        ),
    ]


def train_scratch(train_ds, val_ds, config: Config):
    print("\n" + "=" * 60)
    print("TRAINING: From-Scratch CNN")
    print("=" * 60)

    model = ScratchClassifier(num_classes=config.num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=get_scratch_callbacks(config.output_dir),
    )

    model.save(str(config.output_dir / "scratch_final.keras"))
    print("Saved: scratch_final.keras")
    return model


def train_pretrained_frozen(train_ds, val_ds, config: Config):
    print("\n" + "=" * 60)
    print("TRAINING: Pretrained (frozen backbone — zero-shot baseline)")
    print("=" * 60)

    model = PretrainedClassifier(
        num_classes=config.num_classes,
        image_size=config.image_size,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Only train the classification head, backbone stays frozen
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=get_callbacks("pretrained_frozen", config.output_dir),
    )

    model.save(str(config.output_dir / "pretrained_frozen.keras"))
    print("Saved: pretrained_frozen.keras")
    return model


def train_pretrained_finetuned(train_ds, val_ds, config: Config):
    print("\n" + "=" * 60)
    print("TRAINING: Pretrained + Fine-tuned (unfrozen backbone)")
    print("=" * 60)

    model = PretrainedClassifier(
        num_classes=config.num_classes,
        image_size=config.image_size,
    )

    # Phase 1: train classification head with frozen backbone
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=get_callbacks("finetuned_phase1", config.output_dir),
    )

    # Phase 2: unfreeze last 10 layers, train with lower learning rate
    model.unfreeze(num_layers=10)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.pretrained_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs - 10,
        callbacks=get_callbacks("finetuned_phase2", config.output_dir),
    )

    model.save(str(config.output_dir / "pretrained_finetuned.keras"))
    print("Saved: pretrained_finetuned.keras")
    return model


def main():
    config = Config()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    images, labels = prepare_data(config, samples_per_category=1000)
    (train_scratch_ds, val_scratch_ds), (train_pre_ds, val_pre_ds) = create_tf_datasets(
        images, labels, config
    )

    train_scratch(train_scratch_ds, val_scratch_ds, config)
    train_pretrained_frozen(train_pre_ds, val_pre_ds, config)
    train_pretrained_finetuned(train_pre_ds, val_pre_ds, config)

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE")
    print(f"Models saved to {config.output_dir}/")
    print("Run `python -m src.evaluate` to compare results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
