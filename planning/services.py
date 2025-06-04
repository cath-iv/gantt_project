from collections import defaultdict
from django.db import transaction
from django.db.models import Sum
from .models import Task, ResourceRemains, ProjectProgress, TaskStandards


def calculate_project_progress(project_id):
    with transaction.atomic():
        # 1. Очистка старых данных
        ProjectProgress.objects.filter(project_id=project_id).delete()

        # 2. Получаем нормативы для задач
        standards = {
            ts.task_id: (ts.man_hours, ts.machine_hours)
            for ts in TaskStandards.objects.filter(project_id=project_id)
        }

        # 3. Получаем задачи с ресурсами и создаем словарь
        tasks = Task.objects.filter(project_id=project_id).select_related('resource')
        task_dict = {t.task_id: t for t in tasks}

        # 4. Рассчитываем общий вес и веса по категориям
        total_weight = 0
        category_totals = defaultdict(float)

        for task in tasks:
            man_hours, machine_hours = standards.get(task.task_id, (0, 0))
            task_weight = (man_hours or 0) + (machine_hours or 0)
            total_weight += task_weight
            category_totals[task.work_type] += task_weight

        if total_weight <= 0:
            return 0

        # 5. Получаем фактические выполнения с группировкой по ЗАДАЧЕ (а не типу работ)
        remains = ResourceRemains.objects.filter(
            task__project_id=project_id,
            output__gt=0
        ).values('date', 'task_id').annotate(total_output=Sum('output'))

        # 6. Инициализация структур данных
        daily_stats = defaultdict(lambda: {
            'progress': 0,
            'man_hours': 0,
            'machine_hours': 0,
            'categories': defaultdict(float)
        })

        # 7. Заполнение daily_stats с учетом КАЖДОЙ ЗАДАЧИ отдельно
        for r in remains:
            date_str = r['date'].strftime('%Y-%m-%d')
            task_id = r['task_id']
            output = r['total_output']

            task = task_dict.get(task_id)
            if not task or not task.resource:
                continue

            # Получаем норматив для конкретной задачи
            man_hours_std, machine_hours_std = standards.get(task_id, (0, 0))
            total_std = man_hours_std + machine_hours_std

            if task.resource.quantity > 0:
                ratio = min(output / task.resource.quantity, 1.0)  # Не более 100% на задачу

                # Обновляем статистику
                daily_stats[date_str]['progress'] += ratio * total_std
                daily_stats[date_str]['man_hours'] += ratio * man_hours_std
                daily_stats[date_str]['machine_hours'] += ratio * machine_hours_std
                daily_stats[date_str]['categories'][task.work_type] += ratio * total_std

        # 8. Инициализация кумулятивных значений по категориям
        cumulative_category = {category: 0.0 for category in category_totals}
        records = []
        cumulative_total = 0

        # Сортируем даты в хронологическом порядке
        sorted_dates = sorted(daily_stats.keys())

        for date_str in sorted_dates:
            data = daily_stats[date_str]

            # Общий прогресс
            daily_progress = data['progress'] / total_weight
            cumulative_total += daily_progress

            # Прогресс по категориям (только для существующих категорий)
            for work_type, contribution in data['categories'].items():
                # Пропускаем категории без веса
                if work_type not in category_totals or category_totals[work_type] <= 0:
                    continue

                # Доля выполнения от общего объема категории
                daily_ratio = contribution / category_totals[work_type]
                cumulative_category[work_type] += daily_ratio

            # Создаем запись прогресса
            progress_record = ProjectProgress(
                project_id=project_id,
                date=date_str,
                daily_progress=daily_progress,
                cumulative_progress=cumulative_total,
                man_hours=data['man_hours'],
                machine_hours=data['machine_hours']
            )

            # Устанавливаем значения для всех категорий проекта
            for work_type, cum_value in cumulative_category.items():
                setattr(progress_record, work_type, cum_value)

            records.append(progress_record)

        # 9. Сохранение результатов
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