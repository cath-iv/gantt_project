import numpy as np
import tensorflow as tf
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from planning.ml.models import split_dataset, to_supervised, forecast, Model

category_dict = {
    # AT_1: Основные строительные конструкции (здания)
    'ГИ.НЗ.10 ': 'AT_1',  # ЖБК (балки, плиты)
    'ГИ.НЗ.11 ': 'AT_1',  # Ограждающие конструкции
    'ГИ.НЗ.12 ': 'AT_1',  # Стены/перегородки
    'ГИ.НЗ.13 ': 'AT_1',  # Внутренняя отделка
    'ГИ.НЗ.14 ': 'AT_1',  # Окна/двери

    # AT_2: Технологическое оборудование
    'ГИ.НЗ.20 ': 'AT_2',  # ГПА/сепараторы
    'ГИ.НЗ.21 ': 'AT_2',  # Емкости/блок-боксы
    'ГИ.НЗ.22 ': 'AT_2',  # Прочее оборудование

    # AT_3: Инженерные сети (кроме трубопроводов)
    'ГИ.НЗ.18 ': 'AT_3',  # Трубопроводы (сети)
    'ГИ.НЗ.19 ': 'AT_3',  # ЗРА (сети)
    'ГИ.НЗ.29 ': 'AT_3',  # Воздуховоды
    'ГИ.НЗ.63 ': 'AT_3',  # Кабельные системы

    # AT_4: Земляные работы + фундаменты
    'ГИ.НЗ.2 ': 'AT_4',   # Разработка грунта
    'ГИ.НЗ.3 ': 'AT_4',   # Насыпь
    'ГИ.НЗ.4 ': 'AT_4',   # Сваи
    'ГИ.НЗ.37 ': 'AT_4',  # Обратная засыпка
    'ГИ.НЗ.42 ': 'AT_4',  # Траншеи

    # AT_5: Защитные покрытия
    'ГИ.НЗ.23 ': 'AT_5',  # Гидроизоляция
    'ГИ.НЗ.24 ': 'AT_5',  # АКЗ
    'ГИ.НЗ.25 ': 'AT_5',  # Огнезащита
    'ГИ.НЗ.26 ': 'AT_5',  # Теплоизоляция

    # AT_6: Испытания и контроль
    'ГИ.НЗ.5 ': 'AT_6',   # Испытания грунта
    'ГИ.НЗ.17 ': 'AT_6',  # Контроль стыков
    'ГИ.НЗ.27 ': 'AT_6',  # Гидроиспытания

    # AT_7: Электромонтаж
    'ГИ.НЗ.30 ': 'AT_7',  # Приборы
    'ГИ.НЗ.31 ': 'AT_7',  # Лотки/трубы
    'ГИ.НЗ.32 ': 'AT_7',  # Полки
    'ГИ.НЗ.33 ': 'AT_7',  # Кабели
    'ГИ.НЗ.34 ': 'AT_7',  # Заземление

    # AT_8: Сварка трубопроводов
    'ГИ.НЗ.40 ': 'AT_8',  # Сварка ТТ
    'ГИ.НЗ.41 ': 'AT_8',  # Изоляция стыков
    'ГИ.НЗ.48 ': 'AT_8',  # Секции

    # AT_9: Опоры/мачты
    'ГИ.НЗ.35 ': 'AT_9',  # Прожекторные мачты
    'ГИ.НЗ.36 ': 'AT_9',  # Колодцы

    # AT_10: Дорожные работы
    'ГИ.НЗ.38 ': 'AT_10', # Геотекстиль/плитка
    'ГИ.НЗ.62 ': 'AT_10', # Планировка
    'ГИ.НЗ.77 ': 'AT_10', # Песчаное основание

    # AT_11: Спецработы на трубопроводах
    'ГИ.НЗ.43 ': 'AT_11', # Укладка труб
    'ГИ.НЗ.44 ': 'AT_11', # Засыпка траншей
    'ГИ.НЗ.53 ': 'AT_11', # Защитный слой
    'ГИ.НЗ.54 ': 'AT_11', # Азот
    'ГИ.НЗ.55 ': 'AT_11', # Очистка

    # AT_12: Армирование и металлоконструкции
    'ГИ.НЗ.7 ': 'AT_12',  # Армирование
    'ГИ.НЗ.9 ': 'AT_12',  # Монтаж м/к
    'ГИ.НЗ.67 ': 'AT_12', # Обварка каркасов

    # AT_13: Вспомогательные работы
    'ГИ.НЗ.6 ': 'AT_13',  # ТСГ/ГТМ
    'ГИ.НЗ.46 ': 'AT_13', # Прочее
    'ГИ.НЗ.66 ': 'AT_13', # Транспортировка

    # AT_14: Пусконаладка
    'ГИ.НЗ.49 ': 'AT_14', # Обвязка
    'ГИ.НЗ.51 ': 'AT_14', # Балластировка

    # AT_15: Магистральные трубопроводы
    'ГИ.НЗ.15 ': 'AT_15', # Монтаж ТТ
    'ГИ.НЗ.16 ': 'AT_15', # Монтаж ЗРА
    'ГИ.НЗ.40.6 ': 'AT_15' # Устранение разрывов
}

def get_category(work_type):
    for key in category_dict:
        if key in work_type:
            return category_dict[key]
    return 'Other'

def prepare_data(df, staff, weather):
    staff.drop(['code_project', 'rsrс_id', 'rsrc_name', 'contractor_value', 'speciality_name', 'plannedunits'], axis=1, inplace=True)
    staff['report_date'] = pd.to_datetime(staff['report_date'])
    staff.rename(columns={'report_date': 'Date'}, inplace=True)
    staff['actualunits'] = pd.to_numeric(staff['actualunits'], errors='coerce')
    staff.insert(1, 'Labour', 0)
    staff.insert(2, 'Non-Labour', 0)
    staff = staff.groupby('Date').agg(
      Labour=('actualunits', lambda x: x[staff['rsrc_type'] == 'RT_Labor'].sum()),
      Non_Labour=('actualunits', lambda x: x[staff['rsrc_type'] == 'RT_Nonlabor'].sum())
    ).reset_index()
    weather['Date'] = pd.to_datetime(weather['Date'], format='%d.%m.%Y %H:%M')
    weather.set_index('Date', inplace=True)
    weather['RRR'] = pd.to_numeric(weather['RRR'], errors='coerce')
    weather = weather.resample('D').mean()
    df.drop(['Начало', 'Окончание', 'Плановое количество', 'Spreadsheet Field'], axis=1, inplace=True)
    df['Cathegory'] = df['ИД р-ты'].apply(get_category)
    result = df.groupby('Cathegory').sum()
    result.drop('ИД р-ты', axis=1, inplace=True)
    for col in result.columns[1:]:
      result[col] = (result[col]*100) / result['Кол-во по завершении']
    result.drop(['Кол-во по завершении'], axis=1, inplace=True)
    df_transposed = result.transpose()
    df_transposed = df_transposed.reset_index()
    df_transposed = df_transposed.rename(columns={'index': 'Date'})
    df_transposed['Date'] = pd.to_datetime(df_transposed['Date'], format='mixed')
    df_transposed.rename(columns={'Other': '%_per_day'}, inplace=True)
    column_to_move = df_transposed.pop("%_per_day")
    df_transposed.insert(1, "%_per_day", column_to_move)
    df_transposed.insert(2, 'CUM_%', 0.0)
    for i in range(len(df_transposed)):
      if i == 0:
          df_transposed.loc[i, 'CUM_%'] = df_transposed.loc[i, '%_per_day']
      else:
          df_transposed.loc[i, 'CUM_%'] = df_transposed.loc[i, '%_per_day'] + df_transposed.loc[i - 1, 'CUM_%']

      full_columns = ['AT_1', 'AT_2', 'AT_3', 'AT_4', 'AT_5', 'AT_6', 'AT_7', 'AT_8', 'AT_9', 'AT_10', 'AT_11', 'AT_12', 'AT_13', 'AT_14', 'AT_15']
      for col in full_columns:
          if col not in df_transposed.columns:
              df_transposed[col] = 0.0

    df_transposed = pd.merge(df_transposed, staff, on='Date', how='outer')
    df_transposed = pd.merge(df_transposed, weather, on='Date', how='left')
    for col in df_transposed.columns[3:18]:
        df_transposed[col] = df_transposed[col].cumsum().shift(fill_value=0)
    df_transposed['day_of_year'] = df_transposed['Date'].dt.day_of_year
    start_date = df_transposed['Date'].min()
    df_transposed['days_since_start'] = (df_transposed['Date'] - start_date).dt.days
    df_transposed['datetime'] = pd.to_datetime(df_transposed['Date'], unit='s')
    df_transposed.set_index('datetime', inplace=True)
    df_transposed.drop(columns=['Date'], inplace=True)
    df_transposed['day_of_week'] = df_transposed.index.dayofweek
    df_transposed.fillna(0, inplace=True)
    FIXED_FEATURE_ORDER = [
        'CUM_%', '%_per_day', 'AT_1', 'AT_2', 'AT_3', 'AT_4', 'AT_5',
        'AT_6', 'AT_7', 'AT_8', 'AT_9', 'AT_10', 'AT_11', 'AT_12',
        'AT_13', 'AT_14', 'AT_15', 'Labour', 'Non_Labour', 'RRR',
        'Temp', 'P', 'relative_humidity', 'wind_force', 'day_of_year',
        'days_since_start', 'day_of_week'
    ]
    df_transposed = df_transposed[FIXED_FEATURE_ORDER]
    print("Эксель:", df_transposed.columns)
    return df_transposed

def normalize_projects(projects):
    FIXED_FEATURE_ORDER = [
        'CUM_%', '%_per_day', 'AT_1', 'AT_2', 'AT_3', 'AT_4', 'AT_5',
        'AT_6', 'AT_7', 'AT_8', 'AT_9', 'AT_10', 'AT_11', 'AT_12',
        'AT_13', 'AT_14', 'AT_15', 'Labour', 'Non_Labour', 'RRR',
        'Temp', 'P', 'relative_humidity', 'wind_force', 'day_of_year',
        'days_since_start', 'day_of_week'
    ]

    feature_order = FIXED_FEATURE_ORDER

    common_projects = []
    for project in projects:

        for feature in feature_order:
            if feature not in project.columns:
                project[feature] = 0.0
        project = project[feature_order]
        common_projects.append(project)

    numeric_cols = [col for col in FIXED_FEATURE_ORDER
                   if pd.api.types.is_numeric_dtype(common_projects[0][col])]

    scaler = StandardScaler()
    scaler.fit(pd.concat(common_projects)[numeric_cols])

    normalized_projects = []
    for project in common_projects:
        project_scaled = project.copy()
        project_scaled[numeric_cols] = scaler.transform(project[numeric_cols])
        normalized_projects.append(project_scaled)

    return normalized_projects, scaler, feature_order

def generate_synthetic_project(real_project, noise_level=0.001, targets=None, strong_correlations=None):
    if real_project is None:
        raise ValueError("Реальный проект не может быть None")
    if not isinstance(targets, list):
        raise ValueError("Targets должен быть списком")
    if strong_correlations is None:
        raise ValueError("Не передана корреляционная матрица")

    synthetic = real_project.copy()

    for activity in targets:
        if activity not in strong_correlations:
            continue

        influencing_factors = strong_correlations[activity][
            abs(strong_correlations[activity]) > 0.001
            ].index.tolist()

        for factor in influencing_factors:
            if factor in synthetic.columns:
                corr = strong_correlations.loc[factor, activity]
                noise = np.random.normal(0, noise_level, len(synthetic))
                synthetic[activity] += corr * noise * synthetic[factor]

    return synthetic

def preprocess_to_model(projects_list, n_input):
    FIXED_FEATURE_ORDER = [
        'CUM_%', '%_per_day', 'AT_1', 'AT_2', 'AT_3', 'AT_4', 'AT_5',
        'AT_6', 'AT_7', 'AT_8', 'AT_9', 'AT_10', 'AT_11', 'AT_12',
        'AT_13', 'AT_14', 'AT_15', 'Labour', 'Non_Labour', 'RRR',
        'Temp', 'P', 'relative_humidity', 'wind_force', 'day_of_year',
        'days_since_start', 'day_of_week'
    ]
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    projects = projects_list
    all_features = set()
    for project in projects:
        all_features.update(project.columns)

    common_projects = []
    for project in projects:
        for feature in all_features:
            if feature not in project.columns:
                project[feature] = 0
        project = project[list(all_features)]
        common_projects.append(project)


    print("До нормализации:", projects[0].columns)
    normalized_projects, scaler, all_features = normalize_projects(common_projects)
    print("После нормализации:", normalized_projects[0].columns)

    prepared_data = pd.concat(normalized_projects, axis=0)

    features = ['RRR', 'relative_humidity', 'wind_force', 'Temp']
    targets = [col for col in prepared_data.columns if col.startswith('AT_')]

    missing_features = [f for f in features if f not in prepared_data.columns]
    if missing_features:
        print(f"Предупреждение: Отсутствуют фичи {missing_features}")
        features = [f for f in features if f in prepared_data.columns]

    correlation_matrix = prepared_data[features + targets].corr()
    strong_correlations = correlation_matrix[targets].loc[features]

    synthetic_projects = []
    for project in normalized_projects:
        for _ in range(10):
            synthetic = generate_synthetic_project(
                project,
                noise_level=0.001,
                targets=[col for col in FIXED_FEATURE_ORDER if col.startswith('AT_')],
                strong_correlations=strong_correlations
            )

            synthetic = synthetic[FIXED_FEATURE_ORDER]
            synthetic_projects.append(synthetic)

    print(f"Количество синтетических проектов: {len(synthetic_projects)}")
    print("После генерации:", synthetic_projects[0].columns)

    for p in normalized_projects:
        p['is_real'] = 1
    for p in synthetic_projects:
        p['is_real'] = 0

    all_projects = normalized_projects + synthetic_projects
    print("Перед шафлом 0:", all_projects[0].columns)
    np.random.shuffle(all_projects)

    all_train_x, all_train_y, all_test_x, all_test_y = [], [], [], []
    for project in all_projects:
        try:
            train, test = split_dataset(project, n_input)
            train_x, train_y = to_supervised(train, n_input)
            test_x, test_y = to_supervised(test, n_input)

            all_train_x.append(train_x)
            all_train_y.append(train_y)
            all_test_x.append(test_x)
            all_test_y.append(test_y)
        except Exception as e:
            print(f"Error processing project: {str(e)}")
            continue

    try:
        train_x_combined = np.concatenate(all_train_x, axis=0) if all_train_x else np.array([])
        train_y_combined = np.concatenate(all_train_y, axis=0) if all_train_y else np.array([])
        test_x_combined = np.concatenate(all_test_x, axis=0) if all_test_x else np.array([])
        test_y_combined = np.concatenate(all_test_y, axis=0) if all_test_y else np.array([])

        if len(train_x_combined) == 0:
            raise ValueError("No training data generated")

        return train_x_combined, train_y_combined, test_x_combined, test_y_combined, scaler
    except Exception as e:
        raise ValueError(f"Error combining data: {str(e)}")