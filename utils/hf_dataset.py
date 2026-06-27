"""
Выборочная загрузка датасета RFUAV с Hugging Face.

Репозиторий: https://huggingface.co/datasets/kitofrank/RFUAV

Спектрограммы для классификации лежат в подкаталоге
`ImageSet-AllDrones-MatlabPipeline` со структурой:

    ImageSet-AllDrones-MatlabPipeline/
        train/<class_name>/*.jpg
        valid/<class_name>/*.jpg

После загрузки файлы раскладываются в формат, совместимый с
`torchvision.datasets.ImageFolder` (ожидается тренером проекта):

    <output_dir>/
        train/<class_name>/*.jpg
        valid/<class_name>/*.jpg
        download_manifest.json

Манифест фиксирует, какие классы и сколько файлов были скачаны.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Sequence

from huggingface_hub import hf_hub_download, list_repo_files

# Идентификатор датасета на Hugging Face Hub.
REPO_ID = "kitofrank/RFUAV"

# Подмножество репозитория со спектрограммами (Matlab pipeline).
DEFAULT_IMAGE_SET = "ImageSet-AllDrones-MatlabPipeline"

# Допустимые расширения изображений при фильтрации файлов репозитория.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Сплиты, которые поддерживает данный image set.
SPLITS = ("train", "valid")


def _available_classes_from_repo_files(
    repo_files: Sequence[str],
    image_set: str = DEFAULT_IMAGE_SET,
) -> list[str]:
    """
    Извлекает имена классов из уже полученного списка файлов репозитория.

    Парсит пути вида `<image_set>/<split>/<class_name>/<file>.jpg`.
    Отдельная функция позволяет не запрашивать список файлов повторно
    внутри `download_subset()`.

    Args:
        repo_files: Полный список путей файлов из `list_repo_files()`.
        image_set: Префикс image set внутри репозитория.

    Returns:
        Отсортированный список уникальных имён классов.
    """
    classes: set[str] = set()
    prefix = f"{image_set}/"
    for path in repo_files:
        if not path.startswith(prefix):
            continue
        parts = Path(path).parts
        if len(parts) < 4:
            continue
        split, class_name = parts[1], parts[2]
        if split in SPLITS and Path(path).suffix.lower() in IMAGE_EXTENSIONS:
            classes.add(class_name)
    return sorted(classes)


def list_available_classes(
    image_set: str = DEFAULT_IMAGE_SET,
    repo_id: str = REPO_ID,
) -> list[str]:
    """
    Возвращает все классы, доступные в указанном image set на Hugging Face.

    Выполняет сетевой запрос к Hub API (~7 с для полного репозитория RFUAV).

    Args:
        image_set: Префикс image set (по умолчанию Matlab pipeline).
        repo_id: ID датасета на Hugging Face.

    Returns:
        Отсортированный список имён классов (37 для DEFAULT_IMAGE_SET).
    """
    return _available_classes_from_repo_files(
        list_repo_files(repo_id, repo_type="dataset"),
        image_set=image_set,
    )


def _class_files(
    class_name: str,
    split: str,
    image_set: str,
    repo_files: Sequence[str],
) -> list[str]:
    """
    Фильтрует файлы репозитория для конкретного класса и сплита.

    Args:
        class_name: Имя класса (должно совпадать с именем папки на Hub).
        split: `train` или `valid`.
        image_set: Префикс image set.
        repo_files: Кэшированный список всех файлов репозитория.

    Returns:
        Отсортированные относительные пути файлов изображений.
    """
    prefix = f"{image_set}/{split}/{class_name}/"
    return sorted(
        path
        for path in repo_files
        if path.startswith(prefix) and Path(path).suffix.lower() in IMAGE_EXTENSIONS
    )


def download_subset(
    output_dir: str | Path,
    classes: Sequence[str] | None = None,
    max_per_class: int | None = None,
    splits: Sequence[str] = SPLITS,
    image_set: str = DEFAULT_IMAGE_SET,
    repo_id: str = REPO_ID,
) -> dict:
    """
    Скачивает выбранные спектрограммы и раскладывает их для обучения.

    Файлы загружаются через `hf_hub_download` в локальный кэш Hub,
    затем копируются в `output_dir` с плоской структурой train/valid.
    Повторный запуск дополняет/перезаписывает файлы с теми же именами.

    Args:
        output_dir: Корневая папка будущего датасета.
        classes: Список классов для загрузки. None — все доступные классы.
        max_per_class: Лимит изображений на класс в каждом сплите.
            None — скачать все файлы класса.
        splits: Какие сплиты загружать (`train`, `valid` или оба).
        image_set: Подкаталог внутри репозитория HF.
        repo_id: ID датасета на Hugging Face.

    Returns:
        Словарь-манифест с полями `classes`, `files`, `output_dir` и др.
        Дублируется в `output_dir/download_manifest.json`.

    Raises:
        ValueError: Если указан несуществующий класс или неподдерживаемый split.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Один запрос списка файлов на всю операцию загрузки.
    repo_files = list_repo_files(repo_id, repo_type="dataset")
    available_classes = _available_classes_from_repo_files(repo_files, image_set=image_set)

    if classes:
        requested = list(classes)
        missing = sorted(set(requested) - set(available_classes))
        if missing:
            raise ValueError(
                "Unknown classes: "
                + ", ".join(missing)
                + ". Use --list-classes to see available class names."
            )
        selected_classes = requested
    else:
        selected_classes = available_classes

    summary = {
        "repo_id": repo_id,
        "image_set": image_set,
        "output_dir": str(output_path.resolve()),
        "classes": selected_classes,
        "splits": list(splits),
        "max_per_class": max_per_class,
        "files": {},
    }

    for class_name in selected_classes:
        summary["files"][class_name] = {}
        for split in splits:
            if split not in SPLITS:
                raise ValueError(f"Unsupported split '{split}'. Use: {', '.join(SPLITS)}")

            files = _class_files(class_name, split, image_set, repo_files)
            if max_per_class is not None:
                # Берём первые N файлов в лексикографическом порядке.
                files = files[:max_per_class]

            split_dir = output_path / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            downloaded = 0

            for remote_path in files:
                # hf_hub_download кладёт файл в ~/.cache/huggingface/hub/.
                local_file = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=remote_path,
                )
                destination = split_dir / Path(remote_path).name
                shutil.copy2(local_file, destination)
                downloaded += 1

            summary["files"][class_name][split] = downloaded
            logging.info("Downloaded %s/%s: %d files", split, class_name, downloaded)

    manifest_path = output_path / "download_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
