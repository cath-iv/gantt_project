from collections import defaultdict
from django.db import transaction
from django.db.models import Sum
from .models import Task, ResourceRemains, ProjectProgress, TaskStandards
import io
from .models import Models
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model

def calculate_project_progress(project_id):
    with transaction.atomic():

        ProjectProgress.objects.filter(project_id=project_id).delete()

        standards = {
            ts.task_id: (ts.man_hours, ts.machine_hours)
            for ts in TaskStandards.objects.filter(project_id=project_id)
        }

        tasks = Task.objects.filter(project_id=project_id).select_related('resource')
        task_dict = {t.task_id: t for t in tasks}

        total_weight = 0
        category_totals = defaultdict(float)

        for task in tasks:
            man_hours, machine_hours = standards.get(task.task_id, (0, 0))
            task_weight = (man_hours or 0) + (machine_hours or 0)
            total_weight += task_weight
            category_totals[task.work_type] += task_weight

        if total_weight <= 0:
            return 0

        remains = ResourceRemains.objects.filter(
            task__project_id=project_id,
            output__gt=0
        ).values('date', 'task_id').annotate(total_output=Sum('output'))


        daily_stats = defaultdict(lambda: {
            'progress': 0,
            'man_hours': 0,
            'machine_hours': 0,
            'categories': defaultdict(float)
        })

        for r in remains:
            date_str = r['date'].strftime('%Y-%m-%d')
            task_id = r['task_id']
            output = r['total_output']

            task = task_dict.get(task_id)
            if not task or not task.resource:
                continue

            man_hours_std, machine_hours_std = standards.get(task_id, (0, 0))
            total_std = man_hours_std + machine_hours_std

            if task.resource.quantity > 0:
                ratio = min(output / task.resource.quantity, 1.0)

                daily_stats[date_str]['progress'] += ratio * total_std
                daily_stats[date_str]['man_hours'] += ratio * man_hours_std
                daily_stats[date_str]['machine_hours'] += ratio * machine_hours_std
                daily_stats[date_str]['categories'][task.work_type] += ratio * total_std

        cumulative_category = {category: 0.0 for category in category_totals}
        records = []
        cumulative_total = 0

        sorted_dates = sorted(daily_stats.keys())

        for date_str in sorted_dates:
            data = daily_stats[date_str]

            daily_progress = data['progress'] / total_weight
            cumulative_total += daily_progress

            for work_type, contribution in data['categories'].items():

                if work_type not in category_totals or category_totals[work_type] <= 0:
                    continue

                daily_ratio = contribution / category_totals[work_type]
                cumulative_category[work_type] += daily_ratio

            progress_record = ProjectProgress(
                project_id=project_id,
                date=date_str,
                daily_progress=daily_progress,
                cumulative_progress=cumulative_total,
                man_hours=data['man_hours'],
                machine_hours=data['machine_hours']
            )

            for work_type, cum_value in cumulative_category.items():
                setattr(progress_record, work_type, cum_value)

            records.append(progress_record)

        ProjectProgress.objects.bulk_create(records)
        return len(records)


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


    model_bytes = io.BytesIO()
    save_model(keras_model, model_bytes)
    model_bytes.seek(0)

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

    model_db = Models.objects.get(id=model_id)
    with io.BytesIO(model_db.model_data) as f:
        keras_model = load_model(f)

    return keras_model, model_db.model_config