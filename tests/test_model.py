import numpy as np
import tensorflow as tf
import pytest

from config import Config
from src.models.scratch import ScratchClassifier
from src.models.pretrained import PretrainedClassifier


@pytest.fixture
def config():
    return Config()


class TestScratchClassifier:
    def test_output_shape(self, config):
        model = ScratchClassifier(num_classes=config.num_classes)
        dummy = tf.random.normal((2, *config.image_size, 3))
        output = model(dummy, training=False)
        assert output.shape == (2, config.num_classes)

    def test_output_sums_to_one(self, config):
        model = ScratchClassifier(num_classes=config.num_classes)
        dummy = tf.random.normal((2, *config.image_size, 3))
        output = model(dummy, training=False)
        sums = tf.reduce_sum(output, axis=1).numpy()
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_training_step(self, config):
        model = ScratchClassifier(num_classes=config.num_classes)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        dummy = tf.random.normal((4, *config.image_size, 3))
        labels = tf.random.uniform((4,), maxval=config.num_classes, dtype=tf.int32)
        history = model.fit(dummy, labels, epochs=1, verbose=0)
        assert history.history["loss"][0] > 0


class TestPretrainedClassifier:
    def test_output_shape(self, config):
        model = PretrainedClassifier(num_classes=config.num_classes, image_size=config.image_size)
        dummy = tf.random.normal((2, *config.image_size, 3))
        output = model(dummy, training=False)
        assert output.shape == (2, config.num_classes)

    def test_output_sums_to_one(self, config):
        model = PretrainedClassifier(num_classes=config.num_classes, image_size=config.image_size)
        dummy = tf.random.normal((2, *config.image_size, 3))
        output = model(dummy, training=False)
        sums = tf.reduce_sum(output, axis=1).numpy()
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_unfreeze(self, config):
        model = PretrainedClassifier(num_classes=config.num_classes, image_size=config.image_size)
        frozen_count = sum(1 for l in model.backbone.layers if not l.trainable)
        model.unfreeze(num_layers=20)
        unfrozen_count = sum(1 for l in model.backbone.layers if not l.trainable)
        assert unfrozen_count < frozen_count
