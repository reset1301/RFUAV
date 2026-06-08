import os
import shutil
import random

# НАСТРОЙКА ПУТЕЙ (Укажите свои папки)
# Папка, куда вы распаковали скачанный архив с Figshare (где лежат папки с дронами)
SOURCE_DIR = r"c:\111\detectors\RFUAV\dataset\Spectrograms" 

# Папка, куда скрипт разложит данные для обучения проекта RFUAV
OUTPUT_DIR = r"c:\111\detectors\RFUAV\dataset" 

# Процент картинок, который пойдет на валидацию (20%)
VALID_SPLIT = 0.2

def split_data():
    train_dir = os.path.join(OUTPUT_DIR, 'train')
    valid_dir = os.path.join(OUTPUT_DIR, 'valid')
    
    # Получаем список всех классов (папок с моделями дронов)
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    for cls in classes:
        class_source_dir = os.path.join(SOURCE_DIR, cls)
        
        # Создаем целевые папки для каждого класса
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, cls), exist_ok=True)
        
        # Список всех файлов картинок в папке класса
        images = [f for f in os.listdir(class_source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Перемешиваем файлы случайным образом для честного разделения
        random.shuffle(images)
        
        # Считаем индекс разделения
        split_idx = int(len(images) * (1 - VALID_SPLIT))
        train_images = images[:split_idx]
        valid_images = images[split_idx:]
        
        # Копируем файлы в папку train
        for img in train_images:
            shutil.copy(os.path.join(class_source_dir, img), os.path.join(train_dir, cls, img))
            
        # Копируем файлы в папку valid
        for img in valid_images:
            shutil.copy(os.path.join(class_source_dir, img), os.path.join(valid_dir, cls, img))
            
        print(f"Класс '{cls}': {len(train_images)} картинок в train, {len(valid_images)} в valid.")

if __name__ == "__main__":
    split_data()
    print("Разделение датасета успешно завершено!")
