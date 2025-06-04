from django.db.models import Max, Min
from datetime import timedelta
import pandas as pd
from django.db.models import F
#import os
#import django
#os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gantt_project.settings')
#django.setup()
from planning.models import ProjectProgress, ResourceRemains, Staff, Weather, ConstructionProject

def get_last_date_with_data(project_id):
    dates = [
        ProjectProgress.objects.filter(project_id=project_id).aggregate(Max('date'))['date__max'],
        ResourceRemains.objects.filter(task__project_id=project_id).aggregate(Max('date'))['date__max'],
        Staff.objects.filter(resource__task__project_id=project_id).aggregate(Max('date__date'))['date__date__max']
    ]
    valid_dates = [d for d in dates if d is not None]
    return max(valid_dates) if valid_dates else None
def convert_to_datetime(df, column='date'):
    if column in df.columns:
        df[column] = pd.to_datetime(df[column])
    return df
def get_project_start_date(project_id):

    first_progress = ProjectProgress.objects.filter(
        project_id=project_id
    ).order_by('date').first()

    if first_progress:
        return first_progress.date
def get_last_7_days_data(project_id, n_input):
    try:
        last_date = get_last_date_with_data(project_id)
        if last_date is None:
            raise ValueError("Для проекта нет данных")

        start_date = last_date - timedelta(days=(n_input - 1))

        progress_data = ProjectProgress.objects.filter(
            project_id=project_id,
            date__range=[start_date, last_date]
        ).values(
            'date',
            'cumulative_progress',
            'daily_progress',
            'man_hours',
            'machine_hours',
            'AT_1', 'AT_2', 'AT_3', 'AT_4', 'AT_5',
            'AT_6', 'AT_7', 'AT_8', 'AT_9', 'AT_10',
            'AT_11', 'AT_12', 'AT_13', 'AT_14', 'AT_15'
        ).order_by('date')

        progress_df = pd.DataFrame.from_records(progress_data)
        progress_df = convert_to_datetime(progress_df)
        progress_df.rename(columns={
            'cumulative_progress': 'CUM_%',
            'daily_progress': '%_per_day'
        }, inplace=True)

        weather_data = Weather.objects.filter(date__range=[start_date, last_date])
        weather_df = pd.DataFrame.from_records(weather_data.values())
        weather_df = convert_to_datetime(weather_df)
        weather_df.rename(columns={
            'temp': 'Temp',
            'pressure': 'P',
            'rel_humidity': 'relative_humidity',
            'wind_power': 'wind_force',
            'rainfall': 'RRR'
        }, inplace=True)
        staff_data = Staff.objects.filter(
            date__date__range=[start_date, last_date]
        ).annotate(
            actual_date=F('date__date')
        ).values('actual_date', 'labour', 'quantity_staff')

        staff_list = []
        for item in staff_data:
            staff_list.append({
                'date': item['actual_date'],
                'Labour': item['quantity_staff'] if item['labour'] else 0,
                'Non_Labour': item['quantity_staff'] if not item['labour'] else 0
            })

        labour_df = pd.DataFrame(staff_list)
        if not labour_df.empty:
            labour_df = labour_df.groupby('date').agg({
                'Labour': 'sum',
                'Non_Labour': 'sum'
            }).reset_index()
        else:
            labour_df = pd.DataFrame(columns=['date', 'Labour', 'Non_Labour'])

        labour_df = convert_to_datetime(labour_df)
        result_df = progress_df.merge(weather_df, on='date', how='left')
        result_df = result_df.merge(labour_df, on='date', how='left')
        result_df.fillna(0, inplace=True)

        at_columns = [f'AT_{i}' for i in range(1, 16)]
        for col in at_columns:
            if col in result_df.columns:
                result_df[col] = result_df[col].shift(1).fillna(0)

        project_start_date = get_project_start_date(project_id)
        project_start_ts = pd.Timestamp(project_start_date)
        result_df['days_since_start'] = (result_df['date'] - project_start_ts).dt.days
        result_df['day_of_week'] = result_df['date'].dt.dayofweek
        result_df['day_of_year'] = result_df['date'].dt.day_of_year
        result_df['datetime'] = pd.to_datetime(result_df['date'])
        result_df.set_index('datetime', inplace=True)
        result_df.drop(columns=['date'], inplace=True)

        for i in range(1, 16):
            col = f'AT_{i}'
            if col not in result_df.columns:
                result_df[col] = 0.0

        FIXED_FEATURE_ORDER = [
            'CUM_%', '%_per_day', 'AT_1', 'AT_2', 'AT_3', 'AT_4', 'AT_5',
            'AT_6', 'AT_7', 'AT_8', 'AT_9', 'AT_10', 'AT_11', 'AT_12',
            'AT_13', 'AT_14', 'AT_15', 'Labour', 'Non_Labour', 'RRR',
            'Temp', 'P', 'relative_humidity', 'wind_force', 'day_of_year',
            'days_since_start', 'day_of_week'
        ]

        for col in FIXED_FEATURE_ORDER:
            if col not in result_df.columns:
                result_df[col] = 0

        result_df = result_df[FIXED_FEATURE_ORDER]
        print("7 дней:", result_df.columns)
        return result_df

    except Exception as e:
        print(f"Ошибка при получении данных: {str(e)}")
        return pd.DataFrame()
def get_full_project_data(project_id):
    try:

        date_range = ProjectProgress.objects.filter(
            project_id=project_id
        ).aggregate(
            min_date=Min('date'),
            max_date=Max('date')
        )

        if not date_range['min_date'] or not date_range['max_date']:
            raise ValueError("Для проекта нет данных")

        start_date = date_range['min_date']
        end_date = date_range['max_date']

        progress_data = ProjectProgress.objects.filter(
            project_id=project_id,
            date__range=[start_date, end_date]
        ).values(
            'date',
            'cumulative_progress',
            'daily_progress',
            'man_hours',
            'machine_hours',
            'AT_1', 'AT_2', 'AT_3', 'AT_4', 'AT_5',
            'AT_6', 'AT_7', 'AT_8', 'AT_9', 'AT_10',
            'AT_11', 'AT_12', 'AT_13', 'AT_14', 'AT_15'
        ).order_by('date')

        progress_df = pd.DataFrame.from_records(progress_data)
        progress_df = convert_to_datetime(progress_df)
        progress_df.rename(columns={
            'cumulative_progress': 'CUM_%',
            'daily_progress': '%_per_day'
        }, inplace=True)
        weather_data = Weather.objects.filter(date__range=[start_date, end_date])
        weather_df = pd.DataFrame.from_records(weather_data.values())
        weather_df = convert_to_datetime(weather_df)
        weather_df.rename(columns={
            'temp': 'Temp',
            'pressure': 'P',
            'rel_humidity': 'relative_humidity',
            'wind_power': 'wind_force',
            'rainfall': 'RRR'
        }, inplace=True)

        staff_data = Staff.objects.filter(
            date__date__range=[start_date, end_date]
        ).annotate(
            actual_date=F('date__date')
        ).values('actual_date', 'labour', 'quantity_staff')

        staff_list = []
        for item in staff_data:
            staff_list.append({
                'date': item['actual_date'],
                'Labour': item['quantity_staff'] if item['labour'] else 0,
                'Non_Labour': item['quantity_staff'] if not item['labour'] else 0
            })

        labour_df = pd.DataFrame(staff_list)
        if not labour_df.empty:
            labour_df = labour_df.groupby('date').agg({
                'Labour': 'sum',
                'Non_Labour': 'sum'
            }).reset_index()
        else:
            labour_df = pd.DataFrame(columns=['date', 'Labour', 'Non_Labour'])

        labour_df = convert_to_datetime(labour_df)
        result_df = progress_df.merge(weather_df, on='date', how='left')
        result_df = result_df.merge(labour_df, on='date', how='left')
        result_df.fillna(0, inplace=True)
        at_columns = [f'AT_{i}' for i in range(1, 16)]
        for col in at_columns:
            if col in result_df.columns:
                result_df[col] = result_df[col].shift(1).fillna(0)

        project_start_date = get_project_start_date(project_id)
        project_start_ts = pd.Timestamp(project_start_date)
        result_df['days_since_start'] = (result_df['date'] - project_start_ts).dt.days
        result_df['day_of_week'] = result_df['date'].dt.dayofweek
        result_df['day_of_year'] = result_df['date'].dt.day_of_year
        result_df['datetime'] = pd.to_datetime(result_df['date'])
        result_df.set_index('datetime', inplace=True)
        result_df.drop(columns=['date'], inplace=True)

        for i in range(1, 16):
            col = f'AT_{i}'
            if col not in result_df.columns:
                result_df[col] = 0.0

        FIXED_FEATURE_ORDER = [
            'CUM_%', '%_per_day', 'AT_1', 'AT_2', 'AT_3', 'AT_4', 'AT_5',
            'AT_6', 'AT_7', 'AT_8', 'AT_9', 'AT_10', 'AT_11', 'AT_12',
            'AT_13', 'AT_14', 'AT_15', 'Labour', 'Non_Labour', 'RRR',
            'Temp', 'P', 'relative_humidity', 'wind_force', 'day_of_year',
            'days_since_start', 'day_of_week'
        ]

        for col in FIXED_FEATURE_ORDER:
            if col not in result_df.columns:
                result_df[col] = 0

        result_df = result_df[FIXED_FEATURE_ORDER]
        print("Весь:", result_df.columns)
        return result_df

    except Exception as e:
        print(f"Ошибка при получении данных: {str(e)}")
        return pd.DataFrame()

