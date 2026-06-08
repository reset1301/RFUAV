import os
from utils.benchmark import Classify_Model

def main():
    # 1. Инициализируем модель классификации ResNet34
    # todo версия модели в конфиг
    print("Инициализация модели ResNet34...")
    test = Classify_Model(cfg='configs/exp0_test.yaml',
                          weight_path='c:/111/detectors/RFUAV/result/best_model.pth')

    # 2. Указываем путь к тестовой картинке-спектрограмме
    # todo configurable
    test_image_path = 'c:/111/detectors/RFUAV/dataset/valid/Spectrogram_Conversion_Type7/data131.jpg'
    
    # Папка, куда скрипт сохранит визуальный результат распознавания
    # todo configurable
    output_folder = './inference_result/'

    # Проверяем физическое наличие файла перед запуском
    if not os.path.exists(test_image_path):
        print(f"Ошибка: Файл {test_image_path} не найден!")
        return

    print(f"Начинаем автоматический инференс для файла: {os.path.basename(test_image_path)}")
    
    # 3. Вызываем встроенный метод авторов репозитория
    # Он сам сделает: чтение -> preprocess -> forward -> сохранение картинки с рамкой/текстом
    test.inference(source=test_image_path, save_path=output_folder)
    
    print("\n" + "="*50)
    print("ИНФЕРЕНС УСПЕШНО ВЫПОЛНЕН!")
    print(f"Результат распознавания сохранен в папку: {os.path.abspath(output_folder)}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
