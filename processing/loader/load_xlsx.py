import os
from pathlib import Path
from dotenv import load_dotenv
import openpyxl

# Загружаем переменные из .env
load_dotenv()
# немного навайбкодил - работает (при следующей декомпозации разобратся!!!)

class PlanogramReader:
    def __init__(self):
        # 1. Определяем корень проекта (поднимаемся на 2 уровня выше от текущего скрипта:
        # processing/loader/load_xlsx.py -> processing/loader -> processing -> корень)
        project_root = Path(__file__).resolve().parent.parent.parent

        # 2. Грузим .env, который лежит в корне
        load_dotenv(project_root / ".env")

        # 3. Достаем путь к папке из .env
        env_dir = os.getenv("PLANOGRAMS_DIR", "data_for_processing/planograms")

        # 4. Формируем финальный абсолютный путь к папке с планограммами
        self.folder_path = (project_root / env_dir).resolve()

    def read_table_to_array(self, file_name):
        file_path = self.folder_path / (file_name + ".xlsx")

        if not file_path.exists():
            raise FileNotFoundError(f"Файл {file_name} не найден")

        wb = openpyxl.load_workbook(file_path, data_only=True)
        sheet = wb.worksheets[0]

        final_data = []
        for row in sheet.iter_rows(values_only=True):
            # Создаем список только из тех ячеек, где есть значение
            # (игнорируем None, пустые строки и пробелы)
            clean_row = [
                str(cell) for cell in row
                if cell is not None and str(cell).strip() != ""
            ]

            # Если в строке нашлось хоть одно значение — добавляем её в общий массив
            if clean_row:
                final_data.append(clean_row)

        wb.close()
        return final_data