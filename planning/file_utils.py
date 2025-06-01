import os
import shutil
import uuid
from django.conf import settings


def save_files_to_temp(files, model_id):
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp', str(model_id))
    os.makedirs(temp_dir, exist_ok=True)

    # Удаляем старые файлы если есть
    for old_file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, old_file))

    # Сохраняем новые файлы с префиксами
    for file_key in files:
        file_obj = files[file_key]
        file_path = os.path.join(temp_dir, f"{file_key}_{file_obj.name}")
        with open(file_path, 'wb+') as destination:
            for chunk in file_obj.chunks():
                destination.write(chunk)
        print(f"Saved {file_key} to {file_path}")

def get_temp_files(model_id):
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp', str(model_id))
    if not os.path.exists(temp_dir):
        return None

    files = {}
    for filename in os.listdir(temp_dir):
        if filename.startswith('project_file_'):
            files['project_file'] = os.path.join(temp_dir, filename)
        elif filename.startswith('staff_file_'):
            files['staff_file'] = os.path.join(temp_dir, filename)
        elif filename.startswith('weather_file_'):
            files['weather_file'] = os.path.join(temp_dir, filename)

    # Проверяем что нашли все три файла
    if len(files) != 3:
        return None

    return files


def delete_temp_files(model_id):
    """Удаляет временные файлы"""
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp', str(model_id))
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)