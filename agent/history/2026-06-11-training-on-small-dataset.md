# История диалога — 2026-06-11

**Проект:** RFUAV  
**Ветка:** `feature/training-on-small-dataset`  
**Тема:** Обучение ResNet18 на небольшой выборке (MacBook Pro M4 Pro)

---

## Сообщение 1 — Пользователь

> Посмотри структуру проекта. Как мне обучить модель resnet18 на небольшой выборке в тестовых целях на macbook pro m4 pro 24gb ram?

---

## Ответ 1 — Ассистент

### Структура проекта RFUAV

Это репозиторий для **классификации и детекции дронов** по RF-спектрограммам.

```
RFUAV/
├── train.py              # Точка входа для обучения (нужно указать путь к .yaml)
├── inference.py          # Инференс
├── benchmark.py          # Оценка модели
├── configs/              # Конфиги обучения (exp1.*, exp3.*, sample.yaml)
├── example/classify/     # Пример конфига ResNet18
├── utils/
│   ├── trainer.py        # CustomTrainer, Basetrainer, model_init_ (ResNet18 и др.)
│   ├── build.py          # Валидация и загрузка YAML-конфига
│   ├── preprocessor.py   # Аугментация, split_images для train/val
│   └── benchmark.py      # Инференс и метрики
├── graphic/              # Конвертация сырых IQ → спектрограммы
├── tools/                # Утилиты для датасета
└── requirements.txt
```

Обучение ResNet18 идёт через `CustomTrainer` и `torchvision.datasets.ImageFolder` — нужна структура папок по классам.

---

### Как обучить ResNet18 на небольшой выборке (MacBook M4 Pro)

#### 1. Окружение

```bash
cd /Users/nik/WORK/Projects/RFUAV
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

В проекте нет готовых изображений — их нужно скачать с [Hugging Face](https://huggingface.co/datasets/kitofrank/RFUAV) (спектрограммы PNG).

#### 2. Подготовить маленький датасет

Для теста достаточно 2–5 классов и по 20–50 картинок на класс.

**Структура (обязательная для `ImageFolder`):**

```
dataset_small/
├── train/
│   ├── DJI AVATA2/
│   │   ├── img001.png
│   │   └── ...
│   └── DJI MINI4 PRO/
│       └── ...
└── valid/          # или val — в конфиге указываете сами
    ├── DJI AVATA2/
    └── DJI MINI4 PRO/
```

Если все картинки лежат в одной папке по классам, можно разбить так:

```python
from utils.preprocessor import split_images

split_images(
    input_path="/path/to/all_images",   # class_name/*.png
    output_path="/path/to/dataset_small",
    train_ratio=0.8
)
```

#### 3. Конфиг для тестового прогона на Mac

Создайте, например, `configs/mac_test_resnet18.yaml`:

```yaml
train: /Users/nik/data/rfuav_small/train
val: /Users/nik/data/rfuav_small/valid
save_path: /Users/nik/data/rfuav_small/runs/resnet18_test

model: resnet18
weights: None
batch_size: 8
image_size: 224        # для теста лучше 224, не 640
num_classes: 2         # число ваших классов
class_names:
  - DJI AVATA2
  - DJI MINI4 PRO
device: cpu            # см. ниже про MPS
num_epochs: 10          # для smoke test хватит 5–10
shuffle: true
lr: 0.0001
optimizer: None
```

Важно:
- `train`, `val`, `save_path` должны существовать до запуска (`save_path` создайте вручную).
- `len(class_names) == num_classes`.
- Имена классов должны совпадать с именами подпапок.

#### 4. Запуск обучения

```python
from utils.trainer import CustomTrainer

trainer = CustomTrainer(cfg='configs/mac_test_resnet18.yaml')
trainer.train()
```

Или напрямую:

```bash
python -c "
from utils.trainer import CustomTrainer
trainer = CustomTrainer(cfg='configs/mac_test_resnet18.yaml')
trainer.train()
"
```

Результаты:
- `best_model.pth` — лучшая модель по val accuracy
- `ResNet_epoch_N.pth` — чекпоинты по эпохам
- `train.log`, `config.yaml` — лог и сохранённый конфиг

#### 5. Устройство на Mac (важный нюанс)

Классификатор **не использует Apple MPS (Metal)**. В `utils/trainer.py` и `utils/build.py` логика такая:

- если нет CUDA → принудительно **CPU**;
- `device: mps` в конфиге сейчас не заработает.

На M4 Pro с 24 GB RAM ResNet18 на маленькой выборке (224×224, batch 8, 2–5 классов) на CPU обычно укладывается в несколько минут на эпоху. Для smoke test это нормально.

#### 6. Рекомендуемые параметры для «тестовых целей»

| Параметр | Значение | Зачем |
|----------|----------|-------|
| `model` | `resnet18` | Лёгкая модель, быстрый прогон |
| `image_size` | `224` | Стандарт для ResNet, меньше памяти |
| `batch_size` | `4`–`8` | Безопасно на Mac |
| `num_epochs` | `5`–`10` | Проверить, что пайплайн работает |
| `num_classes` | `2`–`5` | Маленькая выборка |
| `lr` | `0.0001` | Дефолт проекта |
| `shuffle` | `true` | Перемешивание train |
| `weights` | `None` | Fine-tune с ImageNet-весами (дефолт `pretrained=True`) |

Для реального качества на маленькой выборке имеет смысл аугментация:

```python
from utils.preprocessor import data_augmentation

data_augmentation(
    dataset_path="/path/to/dataset_small",
    output_path="/path/to/dataset_small_aug"
)
```

#### Краткий чеклист

1. `pip install -r requirements.txt`
2. Скачать несколько классов спектрограмм с Hugging Face
3. Разложить в `train/class_name/*.png` и `valid/class_name/*.png`
4. Создать YAML с путями и `device: cpu`
5. Создать `save_path`
6. Запустить `CustomTrainer(cfg=...).train()`

---

## Сообщение 2 — Пользователь

> Все сообщения нашего диалога следует всегда сохранять для истории в ./agent/history/ в формате MarkDown

---

## Ответ 2 — Ассистент

Принято. История диалога сохраняется в `./agent/history/` в формате Markdown. Текущая сессия записана в этот файл.

В дальнейшем каждое новое сообщение будет дополнять историю в том же каталоге.

---

## Сообщение 3 — Пользователь

> Ну кто ты, если не красавчик?

---

## Ответ 3 — Ассистент

Спасибо! Рад помочь. Если понадобится — продолжим с ResNet18, MPS на Mac или чем-то ещё по проекту.

---

## Сообщение 4 — Пользователь

> Итак, давай подготовим конфиг для маленького тестового прогона на cpu, чтобы убедиться, что все работает. Затем, после тестового обучения на cpu, добавим поддержку npu MacBook Pro m4 pro.

---

## Ответ 4 — Ассистент

Подготовлены конфиги, минимальный датасет, поддержка MPS и успешные smoke-тесты на CPU и MPS.

### Что сделано

1. **Конфиги:**
   - `configs/mac_test_resnet18_cpu.yaml` — тест на CPU
   - `configs/mac_test_resnet18_mps.yaml` — тест на Apple MPS (NPU/GPU)

2. **Генератор датасета:** `tools/create_minimal_dataset.py` — 2 класса, 8 train + 4 valid на класс

3. **Поддержка MPS:** `utils/device.py` + обновления в `utils/build.py`, `utils/trainer.py`

4. **Исправления по ходу теста:**
   - `CustomTrainer.train` — убран `@property`, метод стал вызываемым
   - Top-k метрика — не падает при `num_classes < 3`

5. **Обновлён `train.py`** — запуск через `--cfg`

### Команды

```bash
# 1. Датасет (если ещё не создан)
python tools/create_minimal_dataset.py
mkdir -p runs/minimal_test_cpu runs/minimal_test_mps

# 2. CPU smoke test
python train.py --cfg configs/mac_test_resnet18_cpu.yaml

# 3. MPS smoke test (после успешного CPU)
python train.py --cfg configs/mac_test_resnet18_mps.yaml
```

### Результаты smoke-тестов

- **CPU:** 2 эпохи, val accuracy 50% → 50%, `runs/minimal_test_cpu/best_model.pth`
- **MPS:** 2 эпохи, val accuracy 50% → 62.5%, `runs/minimal_test_mps/best_model.pth`, device: `mps`

---

## Сообщение 5 — Пользователь

> Можем добавить возможность загрузки части датасета https://huggingface.co/datasets/kitofrank/RFUAV для подготовки к обучению?

---

## Ответ 5 — Ассистент

Добавлена утилита частичной загрузки спектрограмм с Hugging Face.

### Новые файлы

- `utils/hf_dataset.py` — логика загрузки подмножества
- `tools/download_dataset.py` — CLI
- `configs/mac_test_resnet18_hf.yaml` — пример конфига под загруженные данные
- `huggingface_hub` добавлен в `requirements.txt`

### Использование

```bash
# Список доступных классов (37 шт.)
python tools/download_dataset.py --list-classes

# Скачать 2 класса, по 50 изображений на split
python tools/download_dataset.py \
  --classes "DJI AVATA2" "DJI MINI4 PRO" \
  --max-per-class 50 \
  --output data/rfuav_subset

# Обучение на скачанных данных
mkdir -p runs/rfuav_subset_mps
python train.py --cfg configs/mac_test_resnet18_hf.yaml
```

### Структура после загрузки

```
data/rfuav_subset/
├── train/<class_name>/*.jpg
├── valid/<class_name>/*.jpg
└── download_manifest.json
```

Источник: `ImageSet-AllDrones-MatlabPipeline` из репозитория `kitofrank/RFUAV`.

---

## Сообщение 6 — Пользователь

> Спасибо. На данный момент нужно лишь тщательно задокументировать созданный код в комментариях к коду. На следующем шаге вернемся к дальнейшим действиям по обучению.

---

## Ответ 6 — Ассистент

Добавлена документация в комментариях и docstring во всех созданных и изменённых файлах:

- `utils/device.py` — модуль и функции `mps_is_available`, `resolve_device`
- `utils/hf_dataset.py` — структура HF, все публичные и внутренние функции
- `tools/download_dataset.py` — модуль с примерами CLI
- `tools/create_minimal_dataset.py` — назначение, структура выхода
- `train.py` — режимы classify/detect, примеры запуска
- `utils/build.py` — валидация конфига и нормализация device
- `utils/trainer.py` — device, исправление `CustomTrainer.train()`
- `utils/metrics/base_metric.py` — ограничение Top-k по num_classes
- `configs/mac_test_resnet18_*.yaml` — комментарии с командами подготовки и запуска
