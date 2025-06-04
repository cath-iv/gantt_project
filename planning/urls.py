from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.welcome_view, name='welcome'),
    path('planner/', views.planner_view, name='planner'),

    path('api/projects/', views.create_project, name='create-project'),
    path('api/projects/list/', views.get_projects, name='project-list'),
    path('api/projects/<int:project_id>/', views.project_detail, name='project-detail'),

    path('api/tasks/', views.create_task, name='create-task'),
    path('api/tasks/<int:task_id>/', views.task_detail, name='task-detail'),
    path('api/projects/<int:project_id>/tasks/', views.project_tasks, name='project-tasks'),

    path('api/gantt/<int:project_id>/', views.gantt_data, name='gantt-data'),

    path('api/progress/', views.save_progress, name='save-progress'),
    path('api/weather/', views.get_weather, name='get-weather'),

    path('api/models/<int:model_id>/', views.model_detail, name='model-detail'),
    path('api/projects/<int:project_id>/progress/', views.project_progress, name='project-progress'),
    path('api/projects/<int:project_id>/calculate-progress/', views.calculate_progress, name='calculate-progress'),
    path('api/models/<int:pk>/', views.model_detail),
    path('api/models/<int:model_id>/predict/', views.model_predict, name='predict-model'),
    path('api/models/', views.create_model, name='create-model'),
    path('api/models/<int:model_id>/train/', views.train_model, name='train-model'),
    path('api/models/<int:model_id>/delete/', views.delete_model, name='delete_model'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)