from rest_framework import serializers
from .models import (
    ConstructionProject,
    Resource,
    Task,
    TaskDependency,
    Weather,
    ResourceRemains,
    Staff,
    Models,
)

class ConstructionProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = ConstructionProject
        fields = '__all__'

class ResourceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Resource
        fields = '__all__'


class TaskSerializer(serializers.ModelSerializer):
    planned_amount = serializers.SerializerMethodField()
    remains = serializers.SerializerMethodField()

    class Meta:
        model = Task
        fields = ['task_id', 'description', 'estimated_duration',
                  'start_date', 'end_date', 'work_type', 'resource_id',
                  'planned_amount', 'remains']

    def get_planned_amount(self, obj):
        return obj.resource.quantity

    def get_remains(self, obj):
        return obj.resource.remains

# Сериализатор для TaskDependency
class TaskDependencySerializer(serializers.ModelSerializer):
    task = TaskSerializer(read_only=True)
    dependent_task = TaskSerializer(read_only=True)

    class Meta:
        model = TaskDependency
        fields = '__all__'


class WeatherSerializer(serializers.ModelSerializer):
    class Meta:
        model = Weather
        fields = '__all__'


class ResourceRemainsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResourceRemains
        fields = '__all__'


class StaffSerializer(serializers.ModelSerializer):
    class Meta:
        model = Staff
        fields = '__all__'

class ModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Models
        fields = '__all__'
