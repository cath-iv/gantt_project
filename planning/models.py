from audioop import reverse
from datetime import datetime

from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator


class ConstructionProject(models.Model):
    STATUS_CHOICES = [
        ('planned', 'Запланирован'),
        ('in_progress', 'В процессе'),
        ('completed', 'Завершён'),
        ('cancelled', 'Отменён'),
    ]

    project_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=1000)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='planned')

    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    location = models.CharField(max_length=100, default='Москва')
    def update_dates(self):
        tasks = self.task_set.all()
        if tasks.exists():
            self.start_date = tasks.earliest('start_date').start_date
            self.end_date = tasks.latest('end_date').end_date
            self.save()

    def __str__(self):
        return self.name
    def delete(self, *args, **kwargs):
        Task.objects.filter(project=self).delete()
        super().delete(*args, **kwargs)

    def get_absolute_url(self):
        return reverse('project-detail', kwargs={'project_id': self.project_id})

    class Meta:
        db_table = 'Construction_Project'


class Resource(models.Model):
    TYPE_CHOICES = [
        ('material', 'Материал'),
        ('equipment', 'Оборудование'),
        ('human', 'Персонал'),
    ]

    resource_id = models.AutoField(primary_key=True)
    type = models.CharField(max_length=10, choices=TYPE_CHOICES)
    quantity = models.FloatField(validators=[MinValueValidator(0)])
    remains = models.FloatField(validators=[MinValueValidator(0)])
    #unit = models.CharField(max_length=10, choices=UNIT_CHOICES, default='day')

    class Meta:
        db_table = 'Resource'


class ProjectProgress(models.Model):
    project = models.ForeignKey(ConstructionProject, on_delete=models.CASCADE, db_column='project_id')
    date = models.DateField()
    daily_progress = models.FloatField(default=0)
    cumulative_progress = models.FloatField(default=0)
    man_hours = models.FloatField(default=0)
    machine_hours = models.FloatField(default=0)


    class Meta:
        db_table = 'project_progress'  # Используем lowercase для совместимости
        unique_together = ('project', 'date')
        verbose_name_plural = 'Project Progress'

    def __str__(self):
        return f"{self.project.name} - {self.date}: {self.cumulative_progress:.2%}"
class Task(models.Model):
    WORK_TYPE_CHOICES = [
        ('AT_1', 'Основные строительные конструкции'),
        ('AT_2', 'Технологическое оборудование'),
        ('AT_3', 'Инженерные сети'),
        ('AT_4', 'Земляные работы и фундаменты'),
        ('AT_5', 'Защитные покрытия'),
        ('AT_6', 'Испытания и контроль'),
        ('AT_7', 'Электромонтаж'),
        ('AT_8', 'Сварка трубопроводов'),
        ('AT_9', 'Опоры и мачты'),
        ('AT_10', 'Дорожные работы'),
        ('AT_11', 'Спецработы на трубопроводах'),
        ('AT_12', 'Армирование и металлоконструкции'),
        ('AT_13', 'Вспомогательные работы'),
        ('AT_14', 'Пуско-наладочные работы'),
        ('AT_15', 'Магистральные трубопроводы'),
        ]

    task_id = models.AutoField(primary_key=True)
    description = models.CharField(max_length=1000)
    estimated_duration = models.IntegerField(validators=[MinValueValidator(1)])
    start_date = models.DateField()
    end_date = models.DateField()
    project = models.ForeignKey(ConstructionProject, on_delete=models.CASCADE, db_column='project_id')
    work_type = models.CharField(max_length=20, choices=WORK_TYPE_CHOICES)
    resource = models.ForeignKey(Resource, on_delete=models.CASCADE, db_column='resource_id')

    @property
    def planned_amount(self):
        """Плановое количество из связанного ресурса"""
        return self.resource.quantity

    @property
    def remains(self):
        """Остаток из связанного ресурса"""
        return self.resource.remains

    @property
    def weight(self):
        """Вес задачи в нормочасах"""
        return (self.man_hours or 0) + (self.machine_hours or 0)

    @property
    def progress_per_unit(self):
        """Прогресс на единицу объёма"""
        if self.resource.quantity > 0:
            return self.weight / self.resource.quantity
        return 0

    class Meta:
        db_table = 'Task'
        constraints = [
            models.CheckConstraint(
                check=models.Q(end_date__gte=models.F('start_date')),
                name='task_end_after_start'
            )
        ]

class TaskDependency(models.Model):
    DEPENDENCY_TYPES = [
        ('FS', 'Finish-to-Start'),
        ('SS', 'Start-to-Start'),
        ('FF', 'Finish-to-Finish'),
        ('SF', 'Start-to-Finish'),
    ]

    dependency_id = models.AutoField(primary_key=True)
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name='dependencies', db_column='task_id')
    dependent_task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name='dependent_on', db_column='dependent_task_id')
    type = models.CharField(max_length=2, choices=DEPENDENCY_TYPES)

    class Meta:
        db_table = 'Task_Dependency'
        constraints = [
            models.CheckConstraint(
                check=~models.Q(task=models.F('dependent_task')),
                name='no_self_dependency'
            )
        ]

class Weather(models.Model):
    id_weather = models.AutoField(primary_key=True)
    date = models.DateField(unique=True)
    temp = models.FloatField()
    wind_power = models.FloatField(validators=[MinValueValidator(0)])
    pressure = models.FloatField(validators=[MinValueValidator(0.01)])
    rainfall = models.FloatField(validators=[MinValueValidator(0)])
    rel_humidity = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)])

    class Meta:
        db_table = 'Weather'

class ResourceRemains(models.Model):
    id_remains = models.AutoField(primary_key=True)
    date = models.DateField()  # Просто DateField вместо ForeignKey
    output = models.FloatField(validators=[MinValueValidator(0)])
    resource = models.ForeignKey(Resource, on_delete=models.CASCADE, db_column='resource_id')
    task = models.ForeignKey(Task, on_delete=models.CASCADE, db_column='task_id')

    class Meta:
        db_table = 'Resourse_remains'
        indexes = [
            models.Index(fields=['task', 'date']),
        ]

class Staff(models.Model):
    id_staff = models.AutoField(primary_key=True)
    date = models.ForeignKey(Weather, on_delete=models.CASCADE, to_field='date', db_column='date')
    resource = models.ForeignKey(Resource, on_delete=models.CASCADE, db_column='resource_id')
    quantity_staff = models.IntegerField(validators=[MinValueValidator(0)])
    labour = models.BooleanField()

    class Meta:
        db_table = 'Staff'


from django.db import models
from django.db.models import JSONField  # Используем встроенный JSONField
from datetime import datetime
import tensorflow as tf


class Models(models.Model):
    NEURAL_CHOICES = [
        ('lstm', 'LSTM'),
        ('linear', 'Linear Regression'),
        ('gru', 'GRU'),
    ]

    # Основные параметры модели
    profile_name = models.CharField(max_length=50, verbose_name='Название профиля')
    project = models.ForeignKey(
        'ConstructionProject',
        on_delete=models.CASCADE,
        db_column='project_id',
        verbose_name='Проект'
    )

    # Параметры обучения
    num_epoch = models.IntegerField(verbose_name='Количество эпох')
    batch_size = models.IntegerField(verbose_name='Размер батча')
    slide_window = models.IntegerField(verbose_name='Горизонт прогнозирования (дней)')

    # Тип нейросети
    name_neural = models.CharField(
        max_length=30,
        choices=NEURAL_CHOICES,
        verbose_name='Тип нейросети'
    )

    # Данные модели
    model_data = models.BinaryField(verbose_name='Данные модели')  # Для хранения весов

    # Конфигурация и метрики (используем встроенный JSONField)
    model_config = JSONField(default=dict, verbose_name='Конфигурация модели')
    train_metrics = JSONField(default=dict, verbose_name='Метрики обучения')


    # Системные поля
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Дата создания')
    framework_version = models.CharField(
        max_length=50,
        default=tf.__version__,
        verbose_name='Версия фреймворка'
    )
    model_type = models.CharField(
        max_length=20,
        default='keras',
        verbose_name='Тип модели'
    )

    class Meta:
        db_table = 'Models'
        verbose_name = 'Модель нейронной сети'
        verbose_name_plural = 'Модели нейронных сетей'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.profile_name} ({self.get_name_neural_display()})"
class TaskStandards(models.Model):
    id = models.AutoField(primary_key=True)
    project = models.ForeignKey(
        ConstructionProject,
        on_delete=models.CASCADE,
        db_column='project_id'
    )
    task = models.ForeignKey(
        Task,
        on_delete=models.CASCADE,
        db_column='task_id',
        related_name='task_standards'  # Важно: именно такое имя связи
    )
    man_hours = models.FloatField()
    machine_hours = models.FloatField()
    work_type = models.CharField(max_length=20)

    class Meta:
        db_table = 'Task_Standards'