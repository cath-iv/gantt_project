from rest_framework import viewsets, status, request
from rest_framework.decorators import action
from datetime import datetime, timezone
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from .ml.models import forecast, Model, model_to_bytes, evaluate_model_performance
from .ml.preprocess import prepare_data, preprocess_to_model
from rest_framework.serializers import ModelSerializer
from django.shortcuts import render
from datetime import timedelta
from rest_framework import status
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
import pandas as pd
import io
import tensorflow as tf
from django.utils import timezone
from .models import (
    ConstructionProject,
    Resource,
    Task,
    TaskDependency,
    Weather,
    ResourceRemains,
    Staff,
    Models, ProjectProgress
)
from .serializers import (
    ConstructionProjectSerializer,
    ResourceSerializer,
    TaskSerializer,
    ModelSerializer
)
from .services import calculate_project_progress
from .utils import process_excel


class ConstructionProjectViewSet(viewsets.ModelViewSet):
    queryset = ConstructionProject.objects.all()
    serializer_class = ConstructionProjectSerializer

    def update_dates(self, project):
        """Обновляет даты проекта на основе задач"""
        tasks = project.task_set.all()
        if tasks.exists():
            project.start_date = tasks.earliest('start_date').start_date
            project.end_date = tasks.latest('end_date').end_date
            project.save()


class ResourceViewSet(viewsets.ModelViewSet):
    queryset = Resource.objects.all()
    serializer_class = ResourceSerializer

class TaskViewSet(viewsets.ModelViewSet):
    queryset = Task.objects.all()
    serializer_class = TaskSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)

        task = serializer.instance
        resource = task.resource
        resource.remains -= float(task.estimated_duration)
        resource.save()

        project = task.project
        project.update_dates()

        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

class ModelsViewSet(viewsets.ModelViewSet):

    queryset = Models.objects.all()
    serializer_class = ModelSerializer

    @action(detail=True, methods=['post'])
    def retrain(self, request, pk=None):
        model = self.get_object()
        excel_file = request.FILES.get('excel_file')

        if excel_file:
            df = process_excel(excel_file)
            return Response({'status': 'retrained on excel'})
        return Response({'status': 'retrained on project'})


# API Endpoints
@api_view(['GET'])
def planner_view(request):
    context = {
        'projects': ConstructionProject.objects.all(),
        'models': Models.objects.all(),
        'resources': Resource.objects.all()
    }
    return render(request, 'planning/planner.html', context)


@api_view(['POST'])
def create_project(request):
    """Создание проекта с проверкой"""
    try:
        name = request.data.get('name')
        if not name:
            return Response({'error': 'Название обязательно'}, status=400)

        project = ConstructionProject.objects.create(
            name=name,
            status='planned'
        )

        projects = ConstructionProject.objects.all()
        serializer = ConstructionProjectSerializer(projects, many=True)

        return Response({
            'status': 'success',
            'projects': serializer.data
        }, status=201)
    except Exception as e:
        return Response({'error': str(e)}, status=400)
@api_view(['POST'])
def create_task(request):
    """Создание задачи (упрощенная версия)"""
    try:
        data = request.data
        task = Task.objects.create(
            description=data['description'],
            estimated_duration=data['estimated_duration'],
            start_date=data['start_date'],
            end_date=data.get('end_date'),
            project_id=data['project_id'],
            work_type=data['work_type'],
            resource_id=data['resource_id']
        )
        return Response({'status': 'success', 'task_id': task.task_id})
    except Exception as e:
        return Response({'error': str(e)}, status=400)
@api_view(['PUT', 'DELETE'])
def task_detail(request, task_id):
    task = get_object_or_404(Task, pk=task_id)

    if request.method == 'PUT':
        serializer = TaskSerializer(task, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == 'DELETE':
        task.delete()
        return Response(status=204)
@api_view(['DELETE'])
def delete_project(request, project_id):
    project = get_object_or_404(ConstructionProject, pk=project_id)
    project.delete()
    return Response(status=204)
@api_view(['GET'])
def gantt_data(request, project_id):
    tasks = Task.objects.filter(project_id=project_id).values(
        'id', 'description', 'start_date', 'end_date'
    )
    dependencies = TaskDependency.objects.filter(
        task__project_id=project_id
    ).values('task_id', 'dependent_task_id', 'type')

    return Response({
        'tasks': list(tasks),
        'dependencies': list(dependencies)
    })
@api_view(['GET'])
def project_progress_chart(request, project_id):
    progress_data = (
        ProjectProgress.objects
        .filter(project_id=project_id)
        .order_by('date')
        .values('date', 'cumulative_progress')
    )

    # Рассчитываем плановый прогресс (если нужно)
    #planned_data = calculate_planned_progress(project_id)

    return Response({
        'actual': list(progress_data),
        'planned': planned_data,
        'status': 'success'
    })
@api_view(['POST'])
def update_progress_cache(request, project_id):
    """
    Запускает пересчёт кэша прогресса
    """
    try:
        count = calculate_project_progress(project_id)
        return Response({
            'status': 'success',
            'records_updated': count
        })
    except Exception as e:
        return Response({
            'status': 'error',
            'message': str(e)
        }, status=400)

@api_view(['GET'])
def project_progress(request, project_id):
    try:
        # Пересчитываем прогресс перед получением данных
        from .services import calculate_project_progress
        calculate_project_progress(project_id)  # <-- Добавьте эту строку

        progress_data = (
            ProjectProgress.objects
            .filter(project_id=project_id)
            .order_by('date')
            .values('date', 'cumulative_progress')
        )

        return Response({
            'progress_data': [
                {
                    'date': item['date'].strftime('%Y-%m-%d'),
                    'cumulative_progress': float(item['cumulative_progress'])
                }
                for item in progress_data
            ],
            'predicted': [],  # Пока оставляем пустым
            'status': 'success'
        })
    except Exception as e:
        return Response({'error': str(e), 'status': 'error'}, status=500)

@api_view(['POST'])
def calculate_progress(request, project_id):
    from .services import calculate_project_progress
    try:
        count = calculate_project_progress(project_id)
        return Response({
            'status': 'success',
            'records_created': count
        })
    except Exception as e:
        return Response({
            'status': 'error',
            'message': str(e)
        }, status=400)
@api_view(['GET'])
def get_tasks(request):
    """Получение задач проекта для форм"""
    project_id = request.GET.get('project_id')
    tasks = Task.objects.filter(project_id=project_id).values('task_id', 'description')  # Используем task_id вместо id
    return Response(list(tasks))


@api_view(['GET'])
def get_projects(request):
    """Получение списка всех проектов"""
    projects = ConstructionProject.objects.all()
    serializer = ConstructionProjectSerializer(projects, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def project_tasks(request, project_id):
    """Получение задач проекта"""
    tasks = Task.objects.filter(project_id=project_id)
    return Response({
        'tasks': [
            {
                'task_id': task.task_id,
                'description': task.description,
                'estimated_duration': task.estimated_duration,
                'start_date': task.start_date,
                'end_date': task.end_date,
                'work_type': task.work_type,
                'resource_id': task.resource_id
            }
            for task in tasks
        ]
    })


@api_view(['GET', 'PUT', 'DELETE'])
def task_detail(request, task_id):
    """Управление конкретной задачей"""
    task = get_object_or_404(Task, pk=task_id)

    if request.method == 'GET':
        serializer = TaskSerializer(task)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = TaskSerializer(task, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == 'DELETE':
        task.delete()
        return Response(status=204)


@api_view(['POST'])
def task_list(request):
    """Создание новой задачи"""
    serializer = TaskSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)


@api_view(['GET', 'DELETE'])
def project_detail(request, project_id):
    """Управление конкретным проектом"""
    project = get_object_or_404(ConstructionProject, pk=project_id)

    if request.method == 'GET':
        serializer = ConstructionProjectSerializer(project)
        return Response(serializer.data)

    elif request.method == 'DELETE':
        project.delete()
        return Response(status=204)


@api_view(['GET'])
def get_projects(request):
    """Получение списка всех проектов"""
    projects = ConstructionProject.objects.all()
    serializer = ConstructionProjectSerializer(projects, many=True)
    return Response(serializer.data)


@api_view(['GET', 'DELETE'])
def project_detail(request, project_id):
    """Управление конкретным проектом"""
    project = get_object_or_404(ConstructionProject, pk=project_id)

    if request.method == 'GET':
        serializer = ConstructionProjectSerializer(project)
        return Response(serializer.data)

    elif request.method == 'DELETE':
        project.delete()
        return Response(status=204)


@api_view(['GET'])
def project_tasks(request, project_id):
    """Получение задач конкретного проекта"""
    tasks = Task.objects.filter(project_id=project_id)
    serializer = TaskSerializer(tasks, many=True)
    return Response(serializer.data)


@api_view(['GET', 'PUT', 'DELETE'])
def task_detail(request, task_id):
    """Управление конкретной задачей"""
    task = get_object_or_404(Task, pk=task_id)

    if request.method == 'GET':
        serializer = TaskSerializer(task)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = TaskSerializer(task, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == 'DELETE':
        task.delete()
        return Response(status=204)


@api_view(['GET', 'DELETE'])
def model_detail(request, model_id):
    """Управление конкретной моделью"""
    model = get_object_or_404(Models, pk=model_id)

    if request.method == 'GET':
        serializer = ModelSerializer(model)
        return Response(serializer.data)

    elif request.method == 'DELETE':
        model.delete()
        return Response(status=204)


@api_view(['GET'])
def gantt_data(request, project_id):
    """Данные для диаграммы Ганта"""
    tasks = Task.objects.filter(project_id=project_id).values(
        'task_id', 'description', 'start_date', 'end_date', 'work_type'
    )

    dependencies = TaskDependency.objects.filter(
        task__project_id=project_id
    ).values('task_id', 'dependent_task_id')

    return Response({
        'tasks': list(tasks),
        'dependencies': list(dependencies)
    })

def welcome_view(request):
    return render(request, 'planning/welcome.html')

def planner_view(request):
    context = {
        'projects': ConstructionProject.objects.all(),
        'task': Task.objects.all(),
        'models': Models.objects.all(),
        'resources': Resource.objects.all(),
        'work_type_choices': Task.WORK_TYPE_CHOICES,  # Добавил choices
        'tasks': Task.objects.all().select_related('resource'),
    }
    return render(request, 'planning/planner.html', context)
@api_view(['GET'])
def get_weather(request):
    date = request.GET.get('date')
    if not date:
        return Response({'error': 'Date parameter is required'}, status=400)

    try:
        weather = Weather.objects.filter(date=date).first()
        if weather:
            return Response({
                'temp': weather.temp,
                'wind_power': weather.wind_power,
                'pressure': weather.pressure,
                'rainfall': weather.rainfall,
                'rel_humidity': weather.rel_humidity
            })
        return Response({'error': 'Weather data not found'}, status=404)
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def save_progress(request):
    try:
        data = request.data
        # Создаем запись о выполнении
        progress = ResourceRemains.objects.create(
            date=data['date'],
            output=data['output'],
            resource_id=data['resource_id'],
            task_id=data['task_id']
        )


        # Обновляем остаток ресурса
        resource = Resource.objects.get(pk=data['resource_id'])
        resource.remains -= float(data['output'])
        resource.save()

        return Response({'status': 'success'})
    except Exception as e:
        return Response({'status': 'error', 'message': str(e)}, status=400)

class ModelsViewSet(viewsets.ModelViewSet):
    queryset = Models.objects.all()
    serializer_class = ModelSerializer

# Для получения деталей модели
@api_view(['GET'])
def model_detail(request, model_id):
    try:
        model = Models.objects.get(pk=model_id)
        serializer = ModelSerializer(model)
        return Response(serializer.data)
    except Models.DoesNotExist:
        return Response({'error': 'Model not found'}, status=404)

# Для удаления модели
@api_view(['DELETE'])
def delete_model(request, model_id):
    try:
        model = Models.objects.get(pk=model_id)
        model.delete()
        return Response({'status': 'success'})
    except Models.DoesNotExist:
        return Response({'error': 'Model not found'}, status=404)

def get_latest_project_data(project_id, n_steps):
    """Получает последние данные проекта для прогноза"""
    # Реализация получения данных
    return np.array([...])  # Данные в форме [1, n_steps, n_features]

def generate_future_dates(days):
    """Генерирует даты для прогноза"""
    today = datetime.now()
    return [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days + 1)]


from .file_utils import save_files_to_temp


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def create_model(request):
    try:
        data = request.data
        files = request.FILES

        # Проверяем наличие обязательных файлов
        required_files = ['project_file', 'staff_file', 'weather_file']
        for file_key in required_files:
            if file_key not in files:
                return Response({'error': f'Missing file: {file_key}'}, status=400)

        # Создаем запись модели
        model = Models.objects.create(
            profile_name=data['profile_name'],
            project_id=data['project_id'],
            num_epoch=int(data['num_epoch']),
            batch_size=int(data['batch_size']),
            slide_window=int(data['slide_window']),
            model_type=data['model_type'],
            created_at=timezone.now(),
            framework_version=tf.__version__
        )

        # Сохраняем файлы во временную директорию
        save_files_to_temp(files, model.id)

        return Response({
            'status': 'success',
            'model_id': model.id,
            'message': 'Модель успешно создана. Теперь можно обучить.'
        })
    except Exception as e:
        return Response({'error': str(e)}, status=400)


from .file_utils import get_temp_files, delete_temp_files



@parser_classes([MultiPartParser, FormParser])
@api_view(['POST'])
def train_model(request, model_id):
    try:
        print(f"\n=== Starting training for model {model_id} ===")

        # 1. Получаем модель
        model = Models.objects.get(pk=model_id)
        print(f"Model params - epochs: {model.num_epoch}, batch: {model.batch_size}, window: {model.slide_window}")
        print("Model retrieved:", model.id)

        # 2. Получаем пути к файлам
        file_paths = get_temp_files(model_id)
        print("File paths:", file_paths)

        if not file_paths:
            print("Error: No files found")
            return Response({'error': 'Files not found for training'}, status=400)

        # 3. Чтение файлов
        try:
            print("\nReading files...")
            project_data = pd.read_excel(file_paths['project_file'], engine='openpyxl')
            staff_data = pd.read_excel(file_paths['staff_file'], engine='openpyxl')
            weather_data = pd.read_excel(file_paths['weather_file'], engine='openpyxl')

            print("Files read successfully")
            print("Project data shape:", project_data.shape)
            print("Staff data shape:", staff_data.shape)
            print("Weather data shape:", weather_data.shape)

        except Exception as e:
            print("Error reading files:", str(e))
            return Response({'error': f'Error reading files: {str(e)}'}, status=400)

        # 4. Подготовка данных
        try:
            print("\nPreparing data...")
            prepared_data = prepare_data(project_data, staff_data, weather_data)
            print("Data prepared successfully")
            print("Prepared data columns:", prepared_data.columns.tolist())

        except Exception as e:
            print("Error preparing data:", str(e))
            return Response({'error': f'Data preparation failed: {str(e)}'}, status=400)

        # Преобразование данных для модели
        train_x_combined, train_y_combined, test_x_combined, test_y_combined, scaler = preprocess_to_model(prepared_data, model.slide_window)
        print("Train data prepared successfully")

        model_creator = Model()
        model_instance = model_creator.create_model(
            model_type=model.model_type,
            n_timesteps = train_x_combined.shape[1],
            n_features = train_x_combined.shape[2],
            n_outputs=train_y_combined.shape[1]
        )
        print("Model created")
        # Обучение модели
        model_instance, history = model_creator.train_model(
            model_instance,
            train_x_combined,
            train_y_combined,
            n_outputs=model.slide_window,
            epochs=model.num_epoch,  # Используем сохранённое значение
            batch_size=model.batch_size
        )
        print("Success!")
        # Оценка модели
        test_rmse, test_scores = model_creator.evaluate_model(
            model_instance,
            train_x_combined,
            train_y_combined,
            test_x_combined,
            test_y_combined
        )
        print("Scores", test_rmse, test_scores)
        # Сохранение модели и метрик в БД
        metrics = evaluate_model_performance(model_instance,
                                            train_x_combined,
                                            train_y_combined,
                                            test_x_combined,
                                            test_y_combined)

        try:
            # В train_model:
            buffer = model_to_bytes(model_instance.model)
            model.model_data = buffer

            model.train_metrics = {
                'test_rmse': test_rmse,
                'test_scores': test_scores,
                'test_metrics': {
                   'mae': metrics['mae'],
                   'mape': metrics['mape'],
               }
            }
            model.save()
            delete_temp_files(model_id)
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return Response({'error': f'Model saving failed: {str(e)}'}, status=500)

        return Response({
            'status': 'success',
            'metrics': model.train_metrics,
            'message': 'Модель успешно обучена'
        })
    except Models.DoesNotExist:
        return Response({'error': 'Model not found'}, status=404)
    except Exception as e:
        return Response({'error': str(e)}, status=500)


# Для прогнозирования
@api_view(['POST'])
def model_predict(request, model_id):
    try:
        model = Models.objects.get(pk=model_id)

        # Загрузка модели из БД
        buffer = io.BytesIO(model.model_data)
        loaded_model = tf.keras.models.load_model(buffer)

        # Получение данных для прогноза
        last_7_days = get_latest_project_data(model.project_id, model.slide_window)

        # Прогнозирование
        predicted_values = forecast(loaded_model, last_7_days, n_input=model.slide_window)
        predicted_values = [min(100, max(0, x[0])) for x in predicted_values]  # Ограничение 0-100%

        # Форматирование дат
        dates = generate_future_dates(model.slide_window)

        return Response({
            'status': 'success',
            'forecast': predicted_values,
            'dates': dates
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)

