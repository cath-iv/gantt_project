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
    return df_transposed
def normalize_projects(projects):
    all_data = pd.concat(projects)
    numeric_cols = all_data.select_dtypes(include=['float64', 'int64']).columns
    cols_order = ['CUM_%'] + [col for col in numeric_cols if col != 'CUM_%'] # CUM_% - target
    numeric_cols = cols_order

    scaler = StandardScaler()
    scaler.fit(all_data[numeric_cols])

    normalized_projects = []
    for project in projects:
        scaled_values = scaler.transform(project[numeric_cols])
        df_scaled = pd.DataFrame(scaled_values,
                                columns=numeric_cols,
                                index=project.index)
        normalized_projects.append(df_scaled)
    return normalized_projects, scaler

def generate_synthetic_project(real_project, noise_level=0.001, targets = None, strong_correlations = None):
    synthetic = real_project.copy()
    for activity in targets:
        influencing_factors = strong_correlations[activity][
            abs(strong_correlations[activity]) > 0.001
        ].index.tolist()
        if influencing_factors:
            for factor in influencing_factors:
                corr = strong_correlations.loc[factor, activity]
                noise = np.random.normal(0, noise_level, len(synthetic))
                synthetic[activity] += corr * noise * synthetic[factor]

    return synthetic


def preprocess_to_model(df, n_input):
    # Проверка входных параметров
    if n_input is None or not isinstance(n_input, int) or n_input <= 0:
        raise ValueError(f"Invalid n_input value: {n_input}. Must be positive integer")

    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    # Проверка и подготовка данных
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")

    projects = [df]
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

    # Нормализация
    normalized_projects, scaler = normalize_projects(common_projects)
    prepared_data = pd.concat(normalized_projects, axis=0)

    # Анализ корреляций
    features = ['RRR', 'relative_humidity', 'wind_force', 'Temp']
    targets = [col for col in prepared_data.columns if col.startswith('AT_')]

    # Проверка наличия фичей
    missing_features = [f for f in features if f not in prepared_data.columns]
    if missing_features:
        print(f"Warning: Missing features {missing_features}")
        features = [f for f in features if f in prepared_data.columns]

    correlation_matrix = prepared_data[features + targets].corr()
    strong_correlations = correlation_matrix[targets].loc[features]

    # Генерация синтетических данных
    synthetic_projects = []
    for project in normalized_projects:
        for _ in range(10):
            synthetic = generate_synthetic_project(
                project,
                noise_level=0.001,  # Добавлен отсутствующий параметр
                targets=targets,
                strong_correlations=strong_correlations
            )
            synthetic_projects.append(synthetic)

    print(f"Количество синтетических проектов: {len(synthetic_projects)}")

    # Разметка реальных и синтетических данных
    for p in normalized_projects:
        p['is_real'] = 1
    for p in synthetic_projects:
        p['is_real'] = 0

    all_projects = normalized_projects + synthetic_projects
    np.random.shuffle(all_projects)

    # Подготовка данных для модели
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

    # Объединение данных
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


'''
verbose = 0 # по умолчанию?
n_input = 7 # взяли с фронта, указал пользователь
epochs = 25 # взяли с фронта, указал пользователь
batch_size = 64 # взяли с фронта, указал пользователь
df = pd.read_excel('test2.xlsx') # взяли с фронта, загрузил пользователь
staff = pd.read_excel('staff.xlsx') # взяли с фронта, загрузил пользователь
weather = pd.read_excel('weather.xls', date_format='%d.%m.%Y %H:%M') # взяли с фронта, загрузил пользователь
model_types = ['LSTM', 'GRU', 'RNN', 'Bidirectional', 'CNN', 'TCN'] # на фронте доступные пользователю типы
model_type = ['LSTM'] # взяли с фронта, выбрал пользователь
last_7_days = [] # взяли из разных таблиц и как то преобразовали в нужный формат

df = prepare_data(df, staff, weather)
train_x_combined, train_y_combined, test_x_combined, test_y_combined, scaler = preprocess_to_model(df)
model_creator = Model()
model_instance = model_creator.create_model(model_type=model_type, n_timesteps=n_input, n_features=25)
model, history = model_creator.train_model(model_instance, train_x_combined, train_y_combined, n_outputs=n_input)
test_rmse, test_scores = model_creator.evaluate_model(model_instance, train_x_combined, train_y_combined, test_x_combined, test_y_combined)
predicted_values = forecast(model_instance, last_7_days, n_input=7)
model_id = save_to_db(model_instance.model, scaler, project_id, profile_name, n_steps, n_features, n_outputs, num_epoch, batch_size, slide_window, model_type, train_metrics, framework_version)
loaded_model, loaded_scaler, config = load_from_db(model_id)
'''