from django.urls import path
from . import views

urlpatterns = [
    path('', views.welcome_view, name='welcome'),  # Главная страница — экран приветствия
    path('planner/', views.planner_view, name='planner'),  # Основной интерфейс
    # Проекты
    path('api/projects/', views.create_project, name='create-project'),
    path('api/projects/list/', views.get_projects, name='project-list'),
    path('api/projects/<int:project_id>/', views.project_detail, name='project-detail'),

    # Задачи
    path('api/tasks/', views.create_task, name='create-task'),
    path('api/tasks/<int:task_id>/', views.task_detail, name='task-detail'),
    path('api/projects/<int:project_id>/tasks/', views.project_tasks, name='project-tasks'),

    # Гант
    path('api/gantt/<int:project_id>/', views.gantt_data, name='gantt-data'),

    # Прогресс
    path('api/progress/', views.save_progress, name='save-progress'),
    path('api/weather/', views.get_weather, name='get-weather'),

    # Модели
    path('api/models/', views.create_model, name='model-list'),
    path('api/models/<int:model_id>/', views.model_detail, name='model-detail'),
    path('api/projects/<int:project_id>/progress/', views.project_progress, name='project-progress'),
    path('api/projects/<int:project_id>/calculate-progress/', views.calculate_progress, name='calculate-progress'),
    path('api/models/<int:pk>/train/', views.train_model, name='train-model'),
    path('api/models/<int:pk>/predict/', views.model_predict, name='model-predict'),
]