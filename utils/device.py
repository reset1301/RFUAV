"""
Выбор вычислительного устройства для обучения и инференса.

Модуль централизует логику выбора backend'а PyTorch:
- CUDA (NVIDIA GPU)
- MPS  (Apple Metal Performance Shaders — GPU/NPU на Mac с Apple Silicon)
- CPU  (fallback, если запрошенный backend недоступен)

Используется в `utils.build.check_cfg()` и `utils.trainer.Basetrainer`.
Значение `device` из YAML-конфига проходит через `resolve_device()` до
создания `torch.device`, поэтому в конфиге можно указывать `cpu`, `cuda`,
`mps` или `cuda:0`.
"""

import logging

import torch


def mps_is_available() -> bool:
    """
    Проверяет доступность Apple MPS в текущей установке PyTorch.

    Returns:
        True, если `torch.backends.mps` существует и MPS доступен.
    """
    return bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )


def resolve_device(device: str = "cuda") -> torch.device:
    """
    Преобразует строковый идентификатор устройства в `torch.device`.

    Порядок разрешения:
    1. CUDA — если запрошен `cuda`/`gpu`/`cuda:N` и GPU доступен.
    2. MPS  — если запрошен `mps` и Apple Silicon backend доступен.
    3. CPU  — во всех остальных случаях (включая недоступный backend).

    Args:
        device: Значение из конфига или аргумента CLI. Поддерживаются
            `cpu`, `cuda`, `cuda:0`, `gpu`, `mps`.

    Returns:
        Экземпляр `torch.device`, готовый к передаче в `model.to(device)`.
    """
    requested = str(device).strip().lower()

    # NVIDIA GPU: приоритет, если CUDA явно запрошена и доступна.
    if requested in ("cuda", "gpu") or requested.startswith("cuda:"):
        if torch.cuda.is_available():
            return torch.device(requested if requested.startswith("cuda:") else "cuda")
        logging.warning("CUDA requested but not available, falling back")

    # Apple Silicon: Metal Performance Shaders (M4 Pro и др.).
    if requested == "mps":
        if mps_is_available():
            return torch.device("mps")
        logging.warning("MPS requested but not available, falling back to CPU")

    if requested not in ("cpu", "cuda", "mps", "gpu") and not requested.startswith("cuda:"):
        logging.warning("Unknown device '%s', falling back to CPU", device)

    return torch.device("cpu")
