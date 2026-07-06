"""
Генератор синтетического датасета для smoke-тестов пайплайна обучения.

Создаёт минимальную структуру папок, совместимую с ImageFolder:

    <output>/
        train/class_a/*.png
        train/class_b/*.png
        valid/class_a/*.png
        valid/class_b/*.png

Изображения — однотонные RGB-картинки с разными оттенками по классам.
Этого достаточно, чтобы проверить загрузку данных, forward/backward pass,
сохранение чекпоинтов и работу устройства (CPU/MPS).

Используется конфигами:
- configs/mac_test_resnet18_cpu.yaml
- configs/mac_test_resnet18_mps.yaml

Пример:
    python tools/create_minimal_dataset.py --output data/minimal_test
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def create_class_images(class_dir: Path, class_idx: int, count: int, size: int = 224) -> None:
    """
    Создаёт серию PNG-изображений для одного класса.

    Цвета зависят от индекса класса и номера сэмпла, чтобы классы
    были линейно разделимы даже без аугментаций.

    Args:
        class_dir: Папка класса (будет создана при необходимости).
        class_idx: Числовой индекс класса (0, 1, ...).
        count: Количество изображений.
        size: Сторона квадратного изображения в пикселях.
    """
    class_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        base = (class_idx * 40) % 255
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[:, :, 0] = base
        image[:, :, 1] = (base + i * 10) % 255
        image[:, :, 2] = (base + class_idx * 25) % 255
        Image.fromarray(image).save(class_dir / f"sample_{i:03d}.png")


def create_minimal_dataset(output_dir: str, train_per_class: int = 8, val_per_class: int = 4) -> None:
    """
    Создаёт полный минимальный датасет с двумя классами.

    Args:
        output_dir: Корневая директория датасета.
        train_per_class: Число train-изображений на класс.
        val_per_class: Число valid-изображений на класс.
    """
    root = Path(output_dir)
    classes = ["class_a", "class_b"]

    for split, count in (("train", train_per_class), ("valid", val_per_class)):
        for class_idx, class_name in enumerate(classes):
            create_class_images(root / split / class_name, class_idx, count)

    print(f"Dataset created at: {root.resolve()}")
    print(f"Classes: {', '.join(classes)}")
    print(f"Train images per class: {train_per_class}")
    print(f"Valid images per class: {val_per_class}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a minimal RFUAV test dataset")
    parser.add_argument(
        "--output",
        default="data/minimal_test",
        help="Output dataset directory",
    )
    parser.add_argument("--train-per-class", type=int, default=8)
    parser.add_argument("--val-per-class", type=int, default=4)
    args = parser.parse_args()

    create_minimal_dataset(
        output_dir=args.output,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
    )


if __name__ == "__main__":
    main()
