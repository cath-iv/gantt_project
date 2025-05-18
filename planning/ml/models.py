import io
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from math import sqrt
from numpy import split, array
from pandas import read_csv
import tensorflow as tf
from keras import backend as K


class LSTMForecaster:
    def __init__(self, n_steps=7, n_features=25, n_outputs=7):
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.scaler = StandardScaler()
        self.model = None
        self.history = None

    def _r2(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred, axis=-1)
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        return 1 - ss_res / (ss_tot + K.epsilon())

    def _build_model(self):
        model = Sequential([
            LSTM(200, activation='relu', input_shape=(self.n_steps, self.n_features)),
            RepeatVector(self.n_outputs),
            LSTM(200, activation='relu', return_sequences=True),
            TimeDistributed(Dense(100, activation='elu')),
            TimeDistributed(Dense(1))
        ])
        model.compile(loss='mse', optimizer='adamax', metrics=['mae', 'mape', self._r2])
        return model

    def split_dataset(self, df, train_pers=0.8):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Индекс данных должен быть DatetimeIndex.")

        first_date = df.index.min()
        last_date = df.index.max()

        first_index = df.index.get_loc(first_date)
        last_index = df.index.get_loc(last_date)

        total_windows = (last_index - first_index) // self.n_steps + 1
        train_windows = int(total_windows * train_pers)
        test_windows = total_windows - train_windows

        train_start_index = first_index
        train_end_index = train_start_index + train_windows * self.n_steps

        test_start_index = train_end_index
        test_end_index = test_start_index + test_windows * self.n_steps

        train_data = df.iloc[train_start_index:train_end_index].to_numpy()
        test_data = df.iloc[test_start_index:test_end_index].to_numpy()

        if len(train_data) % self.n_steps != 0:
            train_data = train_data[:-(len(train_data) % self.n_steps)]
        if len(test_data) % self.n_steps != 0:
            test_data = test_data[:-(len(test_data) % self.n_steps)]

        train = array(split(train_data, len(train_data) // self.n_steps))
        test = array(split(test_data, len(test_data) // self.n_steps))

        return train, test

    def to_supervised(self, train):
        data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
        X, y = list(), list()
        in_start = 0
        for _ in range(len(data)):
            in_end = in_start + self.n_steps
            out_end = in_end + self.n_outputs
            if out_end <= len(data):
                X.append(data[in_start:in_end, :])
                y.append(data[in_end:out_end, 0])
            in_start += 1
        return np.array(X), np.array(y)

    def train(self, X, y, epochs=50, batch_size=32, verbose=1):
        self.model = self._build_model()
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=verbose
        )
        return self.history

    def evaluate(self, train_x, train_y, test_x, test_y):
        train_predictions = self.model.predict(train_x, verbose=0)
        train_mae = mean_absolute_error(train_y.flatten(), train_predictions.flatten())
        train_r2 = r2_score(train_y.flatten(), train_predictions.flatten())
        train_mape = mean_absolute_percentage_error(train_y.flatten(), train_predictions.flatten())

        print(f'Обучающая выборка, MAE: {train_mae:.4f}')
        print(f'Обучающая выборка, R²: {train_r2:.4f}')
        print(f'Обучающая выборка, MAPE: {train_mape:.4f}')

        test_predictions = self.model.predict(test_x, verbose=0)
        test_mae = mean_absolute_error(test_y.flatten(), test_predictions.flatten())
        test_r2 = r2_score(test_y.flatten(), test_predictions.flatten())
        test_mape = mean_absolute_percentage_error(test_y.flatten(), test_predictions.flatten())

        print(f'Валидационная выборка, MAE: {test_mae:.4f}')
        print(f'Валидационная выборка, R²: {test_r2:.4f}')
        print(f'Валидационная выборка, MAPE: {test_mape:.4f}')

        test_rmse = np.sqrt(mean_squared_error(test_y.flatten(), test_predictions.flatten()))
        return test_rmse, [test_rmse] * test_y.shape[1]

    def plot_error_vs_epochs(self):
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Ошибка vs Эпохи обучения')
        plt.xlabel('Эпохи')
        plt.ylabel('Значение метрики')
        plt.legend()
        plt.show()

    def generate_synthetic_data(self, real_data, num_synthetic=1000, noise_level=0.05):
        """Генерация синтетических данных на основе реальных"""
        synthetic_data = []

        # Находим сильно коррелирующие признаки
        correlation_matrix = real_data.corr()
        targets = [col for col in real_data.columns if col.startswith('AT_')]
        strong_correlations = correlation_matrix[targets].loc[features]

        for _ in range(num_synthetic):
            synthetic = real_data.copy()
            for activity in targets:
                influencing_factors = strong_correlations[activity][
                    abs(strong_correlations[activity]) > 0.85
                    ].index.tolist()
                if influencing_factors:
                    for factor in influencing_factors:
                        corr = strong_correlations.loc[factor, activity]
                        noise = np.random.normal(0, noise_level, len(synthetic))
                        synthetic[activity] += corr * noise * synthetic[factor]
            synthetic_data.append(synthetic)

        return synthetic_data

    def prepare_training_data(self, projects, n_input=None, train_pers=0.8):
        """Подготовка данных для обучения из нескольких проектов"""
        if n_input is None:
            n_input = self.n_steps

        # Нормализация данных
        all_data = pd.concat(projects)
        self.scaler.fit(all_data)
        normalized_projects = [pd.DataFrame(self.scaler.transform(project),
                                            columns=project.columns, index=project.index)
                               for project in projects]

        # Генерация синтетических данных
        synthetic_projects = self.generate_synthetic_data(all_data)

        # Объединение реальных и синтетических данных
        all_projects = normalized_projects + synthetic_projects

        # Подготовка последовательностей
        all_train_x, all_train_y, all_test_x, all_test_y = [], [], [], []
        for project in all_projects:
            train, test = self.split_dataset(project, train_pers)
            train_x, train_y = self.to_supervised(train)
            test_x, test_y = self.to_supervised(test)
            all_train_x.append(train_x)
            all_train_y.append(train_y)
            all_test_x.append(test_x)
            all_test_y.append(test_y)

        # Конкатенация всех данных
        train_x_combined = np.concatenate(all_train_x, axis=0)
        train_y_combined = np.concatenate(all_train_y, axis=0)
        test_x_combined = np.concatenate(all_test_x, axis=0)
        test_y_combined = np.concatenate(all_test_y, axis=0)

        return train_x_combined, train_y_combined, test_x_combined, test_y_combined

    def plot_daily_mape(self, actual, predicted):
        days = actual.shape[1]
        mape_per_day = []
        for day in range(days):
            mape = mean_absolute_percentage_error(actual[:, day], predicted[:, day]) * 100
            mape_per_day.append(mape)

        plt.plot(range(1, days + 1), mape_per_day, marker='o')
        plt.xlabel('День предсказания')
        plt.ylabel('MAPE (%)')
        plt.title('Ошибка модели по дням')
        plt.grid()
        plt.show()

    def forecast(self, history):
        data = np.array(history)
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        input_x = data[-self.n_steps:, :]
        input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
        yhat = self.model.predict(input_x, verbose=0)
        return yhat[0]

    def save_to_db(self, project_id, profile_name):
        # Сериализация scaler
        scaler_bytes = io.BytesIO()
        pickle.dump(self.scaler, scaler_bytes)
        scaler_bytes.seek(0)

        # Создание записи модели
        model = Models.objects.create(
            project_id=project_id,
            profile_name=profile_name,
            name_neural='lstm',
            model_data=self.model_to_bytes(),
            scaler_data=scaler_bytes.getvalue(),
            model_config={
                'n_steps': self.n_steps,
                'n_features': self.n_features
            }
        )
        return model.id

    @classmethod
    def load_from_db(cls, model_id):
        """Загрузка модели из базы данных"""
        from ..models import Models

        db_model = Models.objects.get(id=model_id)
        model_bytes = io.BytesIO(db_model.model_data)
        model_bytes.seek(0)

        scaler_bytes = io.BytesIO(db_model.scaler_data)
        scaler_bytes.seek(0)
        scaler_params = np.load(scaler_bytes, allow_pickle=True).item()

        forecaster = cls(
            n_steps=db_model.model_config['n_steps'],
            n_features=db_model.model_config['n_features'],
            n_outputs=db_model.model_config['n_outputs']
        )

        forecaster.model = tf.keras.models.load_model(model_bytes)
        forecaster.scaler = StandardScaler()
        forecaster.scaler.mean_ = scaler_params['mean']
        forecaster.scaler.scale_ = scaler_params['scale']
        forecaster.scaler.var_ = scaler_params['var']

        return forecaster