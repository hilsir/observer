import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

class Identification:
    def __init__(self, path_to_model=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not path_to_model:
            raise ValueError("Ошибка: модель по расположению не найдена в .env")

        if not path_to_model:
            raise ValueError("Ошибка: модель по расположению не найдена в .env")

        # Старое поведение (путь от корня проекта, начинающийся с "/") оставлено как
        # запасной вариант для обратной совместимости.
        if Path(path_to_model).is_absolute():
            model_full_path = Path(path_to_model).resolve()
        else:
            # Ищем корень проекта — поднимаемся вверх от этого файла, пока не найдём папку models/.
            # Не завязываемся на то, сколько уровней вложенности у этого файла — оно может меняться.
            project_root = Path(__file__).resolve().parent
            while not (project_root / "models").exists() and project_root != project_root.parent:
                project_root = project_root.parent

            model_full_path = (project_root / path_to_model.lstrip("/\\")).resolve()

        print(f"--- Устройство: {self.device} ---")
        print(f"--- Файл модели: {model_full_path} ---")

        if not model_full_path.exists():
            raise FileNotFoundError(f"Файл не найден! Проверь путь: {model_full_path}")

        # Загрузка сохранёной модели на device
        checkpoint = torch.load(model_full_path, map_location=self.device)
        # Извлекаем список названий классов
        self.classes = checkpoint['classes']

        # Берем архитектуру ResNet50
        self.model = models.resnet50()

        # Меняем последний полносвязный слой на колчество классов из загруженой модели
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.classes))

        # Загружаем в архитектуру уже обученные веса
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Переносим модель на device и переводим в режим оценки .eval()
        self.model = self.model.to(self.device).eval()

        # Пайплайн подготовки картинки перед тем, как отдать её нейросети
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  #224 (стандарт для ResNet)
            transforms.ToTensor(),  # Превращаем в массив чисел (тензор) от 0 до 1
            # Нормализация по среднему и отклонению (стандарт ImageNet)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        try:
            # Открываем изображение и принудительно переводим в RGB (чтобы не было проблем с PNG/RGBA)
            img = Image.open(image_path).convert('RGB')

            # Применяем препроцессинг и добавляем размерность батча (unsqueeze(0))
            # т.к ожидает пак изображений, но сейчас одно
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

            # Отключаем расчет градиентов (режим обучения)
            with torch.no_grad():
                outputs = self.model(img_tensor)  # Прогон через нейросеть (Raw logits)
                # Выравниваем выход вероятность (от 0 до 1), сумма которых равна 1
                probs = F.softmax(outputs, dim=1)[0]

            # Берем Топ-1 наиболее вероятный вариант
            top_probs, top_idxs = torch.topk(probs, 1)

            # print(f"\n🔍 Файл: {os.path.basename(image_path)}")

            _name_class = self.classes[top_idxs[0]]  # Имя класса
            _confidence = top_probs[0].item() * 100  # Уверенность в процентах

            return _name_class,_confidence

        except Exception as e:
            print(f"❌ Ошибка при обработке изображения: {e}")
            return None

# if __name__ == "__main__":
#     model_path = os.getenv("MODEL_10104443_PATH")
#     predictor = Identification(model_path)
#     # Тестовый запуск на картинке 11.png
#     target_image = "../images_goods_test/11.png"
#     name,confidence = predictor.predict(target_image)
#     print(f"{1}. {name} — {confidence:.2f}%")