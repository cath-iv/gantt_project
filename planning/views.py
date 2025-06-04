import tempfile
import traceback
from venv import logger
from rest_framework import viewsets, status, request
from rest_framework.decorators import action
from datetime import timezone
from .file_utils import get_temp_files, delete_temp_files
from .file_utils import save_files_to_temp
from .ml.db_preprocess import get_last_7_days_data, get_full_project_data
from .ml.models import forecast, Model, model_to_bytes, evaluate_model_performance, load_from_db
from .ml.preprocess import prepare_data, preprocess_to_model
from rest_framework.serializers import ModelSerializer
from django.shortcuts import render
from rest_framework import status
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
import pandas as pd
import tensorflow as tf
from django.utils import timezone
import joblib
from io import BytesIO
import numpy as np
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
    #planned_data = calculate_planned_progress(project_id)
    return Response({
        'actual': list(progress_data),
       # 'planned': planned_data,
        'status': 'success'
    })
@api_view(['POST'])
def update_progress_cache(request, project_id):
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
        from .services import calculate_project_progress
        calculate_project_progress(project_id)

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
            'predicted': [],
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

    project_id = request.GET.get('project_id')
    tasks = Task.objects.filter(project_id=project_id).values('task_id', 'description')  # Используем task_id вместо id
    return Response(list(tasks))

@api_view(['GET'])
def get_projects(request):

    projects = ConstructionProject.objects.all()
    serializer = ConstructionProjectSerializer(projects, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def project_tasks(request, project_id):

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

    serializer = TaskSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)


@api_view(['GET', 'DELETE'])
def project_detail(request, project_id):

    project = get_object_or_404(ConstructionProject, pk=project_id)

    if request.method == 'GET':
        serializer = ConstructionProjectSerializer(project)
        return Response(serializer.data)

    elif request.method == 'DELETE':
        project.delete()
        return Response(status=204)


@api_view(['GET'])
def get_projects(request):

    projects = ConstructionProject.objects.all()
    serializer = ConstructionProjectSerializer(projects, many=True)
    return Response(serializer.data)


@api_view(['GET', 'DELETE'])
def project_detail(request, project_id):

    project = get_object_or_404(ConstructionProject, pk=project_id)

    if request.method == 'GET':
        serializer = ConstructionProjectSerializer(project)
        return Response(serializer.data)

    elif request.method == 'DELETE':
        project.delete()
        return Response(status=204)


@api_view(['GET'])
def project_tasks(request, project_id):

    tasks = Task.objects.filter(project_id=project_id)
    serializer = TaskSerializer(tasks, many=True)
    return Response(serializer.data)


@api_view(['GET', 'PUT', 'DELETE'])
def task_detail(request, task_id):

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

    model = get_object_or_404(Models, pk=model_id)

    if request.method == 'GET':
        serializer = ModelSerializer(model)
        return Response(serializer.data)

    elif request.method == 'DELETE':
        model.delete()
        return Response(status=204)


@api_view(['GET'])
def gantt_data(request, project_id):

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

        progress = ResourceRemains.objects.create(
            date=data['date'],
            output=data['output'],
            resource_id=data['resource_id'],
            task_id=data['task_id']
        )

        resource = Resource.objects.get(pk=data['resource_id'])
        resource.remains -= float(data['output'])
        resource.save()

        return Response({'status': 'success'})
    except Exception as e:
        return Response({'status': 'error', 'message': str(e)}, status=400)

class ModelsViewSet(viewsets.ModelViewSet):
    queryset = Models.objects.all()
    serializer_class = ModelSerializer

@api_view(['GET'])
def model_detail(request, model_id):
    try:
        model = Models.objects.get(pk=model_id)
        serializer = ModelSerializer(model)
        return Response(serializer.data)
    except Models.DoesNotExist:
        return Response({'error': 'Model not found'}, status=404)


@api_view(['DELETE'])
def delete_model(request, model_id):
    try:
        model = Models.objects.get(pk=model_id)
        model.delete()
        return Response({'status': 'success'})
    except Models.DoesNotExist:
        return Response({'error': 'Model not found'}, status=404)

def generate_future_dates(last_known_date, days):
    from datetime import timedelta
    return [(last_known_date + timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(1, days+1)]

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def create_model(request):
    try:
        data = request.data
        files = request.FILES

        required_files = ['project_file', 'staff_file', 'weather_file']
        for file_key in required_files:
            if file_key not in files:
                return Response({'error': f'Missing file: {file_key}'}, status=400)

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


        save_files_to_temp(files, model.id)

        return Response({
            'status': 'success',
            'model_id': model.id,
            'message': 'Модель успешно создана. Теперь можно обучить.'
        })
    except Exception as e:
        return Response({'error': str(e)}, status=400)

@parser_classes([MultiPartParser, FormParser])
@api_view(['POST'])
def train_model(request, model_id):
    try:
        print(f"\n=== Starting training for model {model_id} ===")


        model = Models.objects.get(pk=model_id)
        print(f"Model params - epochs: {model.num_epoch}, batch: {model.batch_size}, window: {model.slide_window}")
        print("Model retrieved:", model.id)

        file_paths = get_temp_files(model_id)
        print("File paths:", file_paths)

        if not file_paths:
            print("Error: No files found")
            return Response({'error': 'Files not found for training'}, status=400)


        try:
            print("\nReading files...")
            project_data = pd.read_excel(file_paths['project_file'], engine='openpyxl')
            staff_data = pd.read_excel(file_paths['staff_file'], engine='openpyxl')
            weather_data = pd.read_excel(file_paths['weather_file'], engine='openpyxl')

            print("Files read successfully")

        except Exception as e:
            print("Error reading files:", str(e))
            return Response({'error': f'Error reading files: {str(e)}'}, status=400)


        try:
            print("\nPreparing data...")
            open_project_data = get_full_project_data(model.project_id)
            prepared_data = [prepare_data(project_data, staff_data, weather_data), open_project_data]
            print("Data prepared successfully")


        except Exception as e:
            print("Error preparing data:", str(e))
            return Response({'error': f'Data preparation failed: {str(e)}'}, status=400)


        train_x_combined, train_y_combined, test_x_combined, test_y_combined, scaler = preprocess_to_model(prepared_data, model.slide_window)
        print("Train data prepared successfully")
        feature_names = prepared_data[0].columns.tolist()
        model_config = {
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'target_feature': 'CUM_%'
        }
        model_creator = Model()
        model_instance = model_creator.create_model(
            model_type=model.model_type,
            n_timesteps = train_x_combined.shape[1],
            n_features = train_x_combined.shape[2],
            n_outputs=train_y_combined.shape[1]
        )
        print("Model created")

        model_instance, history = model_creator.train_model(
            model_instance,
            train_x_combined,
            train_y_combined,
            #n_outputs=model.slide_window,
            n_outputs=train_y_combined.shape[1],
            epochs=model.num_epoch,
            batch_size=model.batch_size
        )
        print("Success!")

        test_rmse, test_scores = model_creator.evaluate_model(
            model_instance,
            train_x_combined,
            train_y_combined,
            test_x_combined,
            test_y_combined
        )
        print("Scores", test_rmse, test_scores)

        metrics = evaluate_model_performance(model_instance,
                                            train_x_combined,
                                            train_y_combined,
                                            test_x_combined,
                                            test_y_combined)

        try:
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
            scaler_buffer = BytesIO()
            joblib.dump(scaler, scaler_buffer)
            scaler_bytes = scaler_buffer.getvalue()
            model.scaler_data = scaler_bytes
            model.model_config = model_config

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

def calculate_mape(actual, predicted):

    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    if sum(mask) == 0:
        return None
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

@api_view(['POST'])
def model_predict(request, model_id):
    try:

        loaded_model, scaler, model_config = load_from_db(model_id)
        model = Models.objects.get(pk=model_id)
        last_data = get_last_7_days_data(model.project_id, model.slide_window)
        feature_names = model_config.get('feature_names', [])
        if not feature_names:
            raise ValueError("Feature names not found in model config")
        if 'is_real' not in feature_names:
            feature_names.append('is_real')
        for feature in feature_names:
            if feature not in last_data.columns:
                if feature == 'is_real':
                    last_data[feature] = 1
                else:
                    last_data[feature] = 0.0
        if 'is_real' not in last_data.columns:
            last_data['is_real'] = 1

        X = last_data[feature_names]

        numeric_features = [f for f in feature_names if f != 'is_real']
        X[numeric_features] = scaler.transform(X[numeric_features])

        X = X.values.reshape((1, model.slide_window, -1))

        if X.shape[2] != loaded_model.input_shape[-1]:
            raise ValueError(
                f"Shape mismatch: Model expects {loaded_model.input_shape[-1]} features, "
                f"got {X.shape[2]}. Features: {feature_names}"
            )

        horizon = model.slide_window
        predictions = []
        current_data = X.copy()

        for _ in range(horizon):

            pred = loaded_model.predict(current_data)

            if pred.ndim == 3:
                pred_value = pred[0, 0, 0]
            else:
                pred_value = pred[0, 0]
            predictions.append(pred_value)


            current_data = np.roll(current_data, -1, axis=1)
            target_idx = feature_names.index(model_config['target_feature'])
            current_data[0, -1, target_idx] = pred_value

        dummy = np.zeros((len(predictions), len(scaler.feature_names_in_)))
        target_scaler_idx = list(scaler.feature_names_in_).index(model_config['target_feature'])
        dummy[:, target_scaler_idx] = predictions
        predictions = scaler.inverse_transform(dummy)[:, target_scaler_idx]

        predictions = np.clip(predictions, 0, 100).tolist()
        last_date = pd.to_datetime(last_data.index[-1]).date()
        dates = generate_future_dates(last_date, horizon)

        return Response({
            'status': 'success',
            'forecast': [round(float(x), 2) for x in predictions],
            'dates': [str(date) for date in dates],
            'features_used': feature_names,
            'target_feature': model_config['target_feature']
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return Response({
            'error': f"Prediction failed: {str(e)}",
            'traceback': traceback.format_exc()
        }, status=500)