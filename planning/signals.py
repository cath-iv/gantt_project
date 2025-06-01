from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from .models import ResourceRemains, Task
from .services import calculate_project_progress
from django.db.models.signals import pre_delete
from .models import Models
from .file_utils import delete_temp_files

@receiver(post_save, sender=ResourceRemains)
@receiver(post_delete, sender=ResourceRemains)
def update_progress_on_remain_change(sender, instance, **kwargs):
    if hasattr(instance, 'task'):
        calculate_project_progress(instance.task.project_id)

@receiver(post_save, sender=Task)
@receiver(post_delete, sender=Task)
def update_progress_on_task_change(sender, instance, **kwargs):
    calculate_project_progress(instance.project_id)

@receiver(pre_delete, sender=Models)
def delete_model_files(sender, instance, **kwargs):
    delete_temp_files(instance.id)