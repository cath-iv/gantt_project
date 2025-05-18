import numpy as np
from rest_framework import viewsets, status, request
from rest_framework.decorators import api_view, action
from rest_framework.decorators import api_view
from datetime import datetime, timedelta

from .ml.preprocess import prepare_data
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
    TaskDependencySerializer,
    WeatherSerializer,
    ResourceRemainsSerializer,
    StaffSerializer,
    ModelsSerializer
)
from .services import calculate_project_progress
from .utils import process_excel, get_weather


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
    serializer_class = ModelsSerializer

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
'''

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


        projects = ConstructionProject.objects.all().values('project_id', 'name')
        return Response({
            'status': 'success',
            'projects': list(projects)
        })
    except Exception as e:
        return Response({'error': str(e)}, status=400)
'''

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny

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


from django.db.models import Sum, F
from datetime import timedelta
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import ConstructionProject, ProjectProgress, Task, ResourceRemains


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

@api_view(['POST'])
def create_model(request):
    try:
        model = Models.objects.create(
            profile_name=request.data['profile_name'],
            project_id=request.data['project_id'],
            num_epoch=request.data['num_epoch'],
            batch_size=request.data['batch_size'],
            slide_window=request.data['slide_window'],
            name_neural=request.data['name_neural']
        )
        if 'excel_file' in request.FILES:
            process_excel(request.FILES['excel_file'])
        return Response({'status': 'success', 'model_id': model.id})
    except Exception as e:
        return Response({'error': str(e)}, status=400)

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


from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import Task, ConstructionProject
from .serializers import TaskSerializer

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
        serializer = ModelsSerializer(model)
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

from django.shortcuts import render

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


from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, JSONParser
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
import pandas as pd
import numpy as np
import json
from .models import Models, ConstructionProject
from .ml.models import LSTMForecaster
from .ml.preprocess import prepare_data
from .serializers import ModelsSerializer


@api_view(['GET'])
def get_model_params(request):
    """GET метод для получения параметров модели по ID"""
    model_id = request.GET.get('model_id')
    if not model_id:
        return Response(
            {'error': 'Необходимо указать model_id'},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        model = get_object_or_404(Models, pk=model_id)
        serializer = ModelsSerializer(model)
        return Response(serializer.data)
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@parser_classes([MultiPartParser, JSONParser])
def set_model_params(request):
    """POST метод для установки параметров модели"""
    try:
        # Получаем параметры из JSON или form-data
        data = request.data.dict() if hasattr(request.data, 'dict') else request.data

        required_params = ['n_steps', 'epochs', 'batch_size', 'project_id']
        if not all(param in data for param in required_params):
            return Response(
                {'error': f'Необходимые параметры: {", ".join(required_params)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Проверяем существование проекта
        get_object_or_404(ConstructionProject, pk=data['project_id'])

        # Создаем или обновляем модель
        model, created = Models.objects.update_or_create(
            project_id=data['project_id'],
            profile_name=data.get('profile_name', 'default'),
            defaults={
                'num_epoch': data['epochs'],
                'batch_size': data['batch_size'],
                'slide_window': data['n_steps'],
                'name_neural': data.get('name_neural', 'lstm'),
                'model_config': {
                    'n_steps': data['n_steps'],
                    'n_features': data.get('n_features', 10),
                    'n_outputs': data.get('n_outputs', 7)
                }
            }
        )

        return Response({
            'status': 'success',
            'model_id': model.id,
            'created': created
        }, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_training_data(request):
    """POST метод для загрузки данных обучения"""
    try:
        # Проверяем наличие всех необходимых файлов
        required_files = ['project', 'staff', 'weather']
        if not all(f'{file}_file' in request.FILES for file in required_files):
            return Response(
                {'error': 'Необходимо загрузить 3 файла: project, staff и weather'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Получаем model_id
        model_id = request.POST.get('model_id')
        if not model_id:
            return Response(
                {'error': 'Необходимо указать model_id'},
                status=status.HTTP_400_BAD_REQUEST
            )

        model = get_object_or_404(Models, pk=model_id)

        # Загружаем файлы
        project_df = pd.read_excel(request.FILES['project_file'])
        staff_df = pd.read_excel(request.FILES['staff_file'])
        weather_df = pd.read_excel(request.FILES['weather_file'])

        # Подготавливаем данные
        prepared_data = prepare_data(project_df, staff_df, weather_df)

        # Сохраняем данные в модель (в реальном проекте лучше сохранять в хранилище)
        model.model_config['last_prepared_data'] = {
            'columns': list(prepared_data.columns),
            'shape': prepared_data.shape,
            'min_date': str(prepared_data.index.min()),
            'max_date': str(prepared_data.index.max())
        }
        model.save()

        return Response({
            'status': 'success',
            'model_id': model_id,
            'data_info': model.model_config['last_prepared_data']
        })

    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from .models import Models
from .ml.models import LSTMForecaster
import pandas as pd
import json



@api_view(['POST'])
def train_model(request, pk):
    try:
        model = get_object_or_404(Models, pk=pk)

        # Проверяем загружены ли файлы данных
        if not all([
            request.FILES.get('project_file'),
            request.FILES.get('staff_file'),
            request.FILES.get('weather_file')
        ]):
            return Response(
                {'error': 'Необходимо загрузить все файлы данных (project, staff, weather)'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Загружаем данные
        project_df = pd.read_excel(request.FILES['project_file'])
        staff_df = pd.read_excel(request.FILES['staff_file'])
        weather_df = pd.read_excel(request.FILES['weather_file'])

        # Подготавливаем данные
        prepared_data = prepare_data(project_df, staff_df, weather_df)

        # Сохраняем информацию о данных в модель
        model.model_config.update({
            'last_prepared_data': {
                'columns': list(prepared_data.columns),
                'shape': prepared_data.shape,
                'min_date': str(prepared_data.index.min()),
                'max_date': str(prepared_data.index.max()),
                'n_features': len(prepared_data.columns)
            }
        })

        # Создаем и обучаем модель
        forecaster = LSTMForecaster(
            n_steps=model.slide_window,
            n_features=len(prepared_data.columns),
            n_outputs=7
        )

        train_x, train_y, test_x, test_y = forecaster.prepare_training_data([prepared_data])
        history = forecaster.train(
            train_x, train_y,
            epochs=model.num_epoch,
            batch_size=model.batch_size
        )

        # Сохраняем модель и метрики
        model.model_data = forecaster.model_to_bytes()
        model.train_metrics = {
            'test_metrics': forecaster.evaluate(test_x, test_y),
            'history': history.history
        }
        model.save()

        return Response({
            'status': 'success',
            'model_id': model.id,
            'metrics': model.train_metrics
        })

    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def model_predict(request, pk):
    try:
        model = get_object_or_404(Models, pk=pk)

        # Создаем временный scaler для этого запроса
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        # Получаем данные проекта
        progress = ProjectProgress.objects.filter(
            project=model.project
        ).order_by('-date')[:model.slide_window]

        if len(progress) < model.slide_window:
            return Response(
                {'error': f'Нужно минимум {model.slide_window} дней данных'},
                status=400
            )

        # Подготавливаем данные
        values = [p.cumulative_progress for p in progress]
        scaled = scaler.fit_transform([[v] for v in values])

        # Прогнозируем
        forecaster = LSTMForecaster.load_from_db(model.id)
        forecast = forecaster.predict(np.array([scaled]))
        forecast = scaler.inverse_transform(forecast)

        return Response({
            'status': 'success',
            'forecast': forecast.flatten().tolist()
        })

    except Exception as e:
        return Response({'error': str(e)}, status=500)