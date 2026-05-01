from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    image_size: tuple = (224, 224)
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    pretrained_learning_rate: float = 2e-5
    validation_split: float = 0.2
    categories: list = field(default_factory=lambda: [
        "Shoes",
        "Watches",
        "Sunglasses",
        "Jeans",
    ])
    seed: int = 42

    @property
    def num_classes(self) -> int:
        return len(self.categories)
