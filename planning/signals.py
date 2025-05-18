from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from .models import ResourceRemains, Task
from .services import calculate_project_progress

@receiver(post_save, sender=ResourceRemains)
@receiver(post_delete, sender=ResourceRemains)
def update_progress_on_remain_change(sender, instance, **kwargs):
    if hasattr(instance, 'task'):
        calculate_project_progress(instance.task.project_id)

@receiver(post_save, sender=Task)
@receiver(post_delete, sender=Task)
def update_progress_on_task_change(sender, instance, **kwargs):
    calculate_project_progress(instance.project_id)