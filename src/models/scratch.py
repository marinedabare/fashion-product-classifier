import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.saving import register_keras_serializable


@register_keras_serializable()
class ScratchClassifier(keras.Model):
    """
    Image classifier built from zero — no pretrained weights.
    Custom CNN that learns everything from our training data.
    """

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

        self.conv1 = layers.Conv2D(32, 3, activation="relu", padding="same")
        self.pool1 = layers.MaxPooling2D(2)

        self.conv2 = layers.Conv2D(64, 3, activation="relu", padding="same")
        self.pool2 = layers.MaxPooling2D(2)

        self.conv3 = layers.Conv2D(128, 3, activation="relu", padding="same")
        self.pool3 = layers.MaxPooling2D(2)

        self.conv4 = layers.Conv2D(256, 3, activation="relu", padding="same")

        self.gap = layers.GlobalAveragePooling2D()

        self.dense1 = layers.Dense(256, activation="relu")
        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(128, activation="relu")
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, x, training=False):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.gap(self.conv4(x))
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.classifier(x)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config
