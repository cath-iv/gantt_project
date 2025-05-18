def calculate_project_progress(project_id):
    from django.db import transaction
    from .models import Task, ResourceRemains, ProjectProgress, TaskStandards

    with transaction.atomic():
        # 1. Очистка старых данных
        ProjectProgress.objects.filter(project_id=project_id).delete()

        # 2. Получаем нормативы для задач
        standards = {
            ts.task_id: (ts.man_hours, ts.machine_hours)
            for ts in TaskStandards.objects.filter(project_id=project_id)
        }

        # 3. Получаем задачи с ресурсами
        tasks = Task.objects.filter(project_id=project_id).select_related('resource')

        # 4. Рассчитываем общий вес
        total_weight = sum(
            (standards.get(task.task_id, (0, 0))[0] or 0) + 
            (standards.get(task.task_id, (0, 0))[1] or 0)
            for task in tasks
        )

        if total_weight <= 0:
            return 0

        # 5. Получаем фактические выполнения
        remains = ResourceRemains.objects.filter(
            task__project_id=project_id,
            output__gt=0
        ).select_related('task')

        # 6. Агрегация по дням
        daily_stats = {}
        for r in remains:
            date_str = r.date.strftime('%Y-%m-%d')
            task = r.task
            task_std = standards.get(task.task_id, (0, 0))

            if date_str not in daily_stats:
                daily_stats[date_str] = {'progress': 0, 'man_hours': 0, 'machine_hours': 0}

            if task.resource.quantity > 0:
                ratio = r.output / task.resource.quantity
                daily_stats[date_str]['progress'] += ratio * (task_std[0] + task_std[1])
                daily_stats[date_str]['man_hours'] += ratio * task_std[0]
                daily_stats[date_str]['machine_hours'] += ratio * task_std[1]

        # 7. Сохранение результатов
        cumulative = 0
        records = []
        for date_str in sorted(daily_stats.keys()):
            daily_progress = daily_stats[date_str]['progress'] / total_weight
            cumulative += daily_progress
            records.append(ProjectProgress(
                project_id=project_id,
                date=date_str,
                daily_progress=daily_progress,
                cumulative_progress=cumulative,
                man_hours=daily_stats[date_str]['man_hours'],
                machine_hours=daily_stats[date_str]['machine_hours']
            ))

        ProjectProgress.objects.bulk_create(records)
        return len(records)


import io
import json
from datetime import datetime
from .models import Models
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model


def save_neural_model(
        profile_name,
        project_id,
        keras_model,
        num_epoch,
        batch_size,
        slide_window,
        name_neural,
        model_config,
        train_metrics
):
    """
    Сохраняет модель Keras в базу данных
    """
    # Сериализация модели в бинарный формат
    model_bytes = io.BytesIO()
    save_model(keras_model, model_bytes)
    model_bytes.seek(0)

    # Создание записи в БД
    model = Models.objects.create(
        profile_name=profile_name,
        project_id=project_id,
        num_epoch=num_epoch,
        batch_size=batch_size,
        slide_window=slide_window,
        name_neural=name_neural,
        model_data=model_bytes.getvalue(),
        model_config=model_config,
        train_metrics=train_metrics,
        framework_version=tf.__version__
    )

    return model.id


def load_neural_model(model_id):
    """
    Загружает модель Keras из базы данных
    """
    model_db = Models.objects.get(id=model_id)

    # Десериализация модели
    with io.BytesIO(model_db.model_data) as f:
        keras_model = load_model(f)

    return keras_model, model_db.model_config