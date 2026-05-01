import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.saving import register_keras_serializable


@register_keras_serializable()
class PretrainedClassifier(keras.Model):
    """
    Image classifier using EfficientNetB0 pretrained on ImageNet.
    Can be used in two modes:
    - Frozen (zero-shot): backbone weights stay fixed
    - Fine-tuned: backbone partially unfrozen and trained on our data
    """

    def __init__(self, num_classes, image_size, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.image_size = image_size

        self.backbone = keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(*image_size, 3),
            pooling="avg",
        )
        self.backbone.trainable = False

        self.dense1 = layers.Dense(256, activation="relu")
        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(128, activation="relu")
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, x, training=False):
        x = self.backbone(x, training=False)
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.classifier(x)

    def unfreeze(self, num_layers=20):
        self.backbone.trainable = True
        for layer in self.backbone.layers[:-num_layers]:
            layer.trainable = False

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "image_size": self.image_size,
        })
        return config
