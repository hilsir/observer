import os
import torch
from ultralytics import YOLO


def setup_environment():
    # Проверяем, есть ли доступ к GPU
    if torch.cuda.is_available():
        brand = torch.cuda.get_device_name(0).lower()

        if 'radeon' in brand or 'gfx' in brand:
            print(f"--- Обнаружена карта AMD ({brand}). Применяем фиксы ROCm ---")
            os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
            # Для AMD на некоторых версиях лучше выключить AMP
            use_amp = False
        else:
            print(f"--- Обнаружена карта NVIDIA ({brand}). Используем CUDA ---")
            use_amp = True

        return use_amp
    return False


def train():
    amp_setting = setup_environment()
    print(amp_setting)
    model = YOLO('yolov8s.pt')

    model.train(
        data='datasets/SKU-110K.yaml',
        epochs=100,  # На такой мощной карте можно смело ставить 100+
        imgsz=1024,  # Оставляем высокое разрешение для мелких объектов SKU-110K
        batch=-1,  # АВТО-РЕЖИМ: YOLO сам определит максимальный батч под твои 24ГБ (будет ~16-24)
        device=0,
        amp=True,  # ОБЯЗАТЕЛЬНО для NVIDIA. Дает х2 скорость и экономит память без потери точности
        workers=8,  # Увеличиваем до 8 (у А5000 обычно мощный многоядерный CPU рядом)
        optimizer='AdamW',  # Самый стабильный и эффективный для этой архитектуры
        exist_ok=True,
        patience=20,  # Если 20 эпох нет улучшений — остановится сам (экономит время сервера)
        cache=True  # Если на сервере много оперативной памяти (RAM), это ускорит чтение картинок
    )


if __name__ == '__main__':
    train()