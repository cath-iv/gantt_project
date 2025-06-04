import io
import pickle
import tempfile
from datetime import timezone
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Conv1D, BatchNormalization, GRU, SimpleRNN, Bidirectional, LSTM, MaxPooling1D, Flatten, Dense, GlobalAveragePooling1D
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from numpy import array, split

from planning.models import Models

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class BaseModel:
    def __init__(self, n_timesteps, n_features, n_outputs):
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.model = Sequential()


    def add_common_layers(self):
        self.model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same',
                     input_shape=(self.n_timesteps, self.n_features)))
        self.model.add(BatchNormalization())

    def compile_model(self, n_outputs):
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(n_outputs, kernel_initializer='zeros'))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
            loss='log_cosh',
            metrics=['mse', MeanAbsoluteError(name='mae'), MeanAbsolutePercentageError()],
            steps_per_execution=16
        )

    def fit(self, train_x, train_y, epochs=1, batch_size=256, verbose=1):
        history = self.model.fit(
            train_x, train_y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=0.2,
            shuffle=False,
            callbacks=[
                ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True, save_weights_only=False),
                EarlyStopping(patience=5, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=2)
            ]
        )
        return history

class GRUModel(BaseModel):
    def build_model(self):
        self.add_common_layers()
        self.model.add(GRU(128, return_sequences=True, dropout=0.3, kernel_regularizer='l2'))
        self.model.add(BatchNormalization())
        self.model.add(GRU(64, dropout=0.2))

class RNNModel(BaseModel):
    def build_model(self):
        self.add_common_layers()
        self.model.add(SimpleRNN(128, return_sequences=True, activation='relu', kernel_regularizer='l2'))
        self.model.add(BatchNormalization())
        self.model.add(SimpleRNN(64))

class BidirectionalModel(BaseModel):
    def build_model(self):
        self.add_common_layers()
        self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
        self.model.add(BatchNormalization())
        self.model.add(Bidirectional(LSTM(32)))

class CNNModel(BaseModel):
    def build_model(self):
        self.model.add(Conv1D(128, kernel_size=5, activation='relu',padding='same', input_shape=(self.n_timesteps, self.n_features)))
        self.model.add(MaxPooling1D(2))
        self.model.add(Conv1D(64, kernel_size=2, activation='relu', dilation_rate=1))
        self.model.add(Flatten())

class TCNModel(BaseModel):
    def build_model(self):
        for i in range(3):
            self.model.add(Conv1D(64 * (2 ** i), kernel_size=3, dilation_rate=2 ** i, padding='same', activation='relu', input_shape=(self.n_timesteps, self.n_features)))
        self.model.add(GlobalAveragePooling1D())

class LSTMModel(BaseModel):
    def build_model(self):
        self.add_common_layers()
        self.model.add(LSTM(128, return_sequences=True, dropout=0.3, kernel_regularizer='l2'))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(64, dropout=0.2))

class Model:
    def __init__(self):
        self.model_classes = {
            'GRU': GRUModel,
            'RNN': RNNModel,
            'Bidirectional': BidirectionalModel,
            'CNN': CNNModel,
            'TCN': TCNModel,
            'LSTM': LSTMModel
        }

    def create_model(self, model_type, n_timesteps, n_features, n_outputs):
        if model_type in self.model_classes:
            model_instance = self.model_classes[model_type](n_timesteps, n_features, n_outputs)
            model_instance.build_model()
            return model_instance
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train_model(self, model, train_x, train_y, n_outputs, epochs=1, batch_size=256, verbose=1):
        model.compile_model(n_outputs)
        history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return model, history

    def evaluate_model(self, model, train_x, train_y, test_x, test_y):
        train_predictions = model.model.predict(train_x, verbose=0)
        train_mae = mean_absolute_error(train_y.flatten(), train_predictions.flatten())
        train_r2 = r2_score(train_y.flatten(), train_predictions.flatten())
        train_mape = mean_absolute_percentage_error(train_y.flatten(), train_predictions.flatten())

        print(f'Обучающая выборка, MAE: {train_mae:.4f}')
        print(f'Обучающая выборка, R²: {train_r2:.4f}')
        print(f'Обучающая выборка, MAPE: {train_mape:.4f}')

        test_predictions = model.model.predict(test_x, verbose=0)
        test_mae = mean_absolute_error(test_y.flatten(), test_predictions.flatten())
        test_r2 = r2_score(test_y.flatten(), test_predictions.flatten())
        test_mape = mean_absolute_percentage_error(test_y.flatten(), test_predictions.flatten())

        print(f'Валидационная выборка, MAE: {test_mae:.4f}')
        print(f'Валидационная выборка, R²: {test_r2:.4f}')
        print(f'Валидационная выборка, MAPE: {test_mape:.4f}')

        test_rmse = np.sqrt(mean_squared_error(test_y.flatten(), test_predictions.flatten()))
        return test_rmse, [test_rmse] * test_y.shape[1]

def split_dataset(df, n_input, train_pers=0.8):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Индекс данных должен быть DatetimeIndex.")

    first_date = df.index.min()
    last_date = df.index.max()

    first_index = df.index.get_loc(first_date)
    last_index = df.index.get_loc(last_date)

    total_windows = (last_index - first_index) // n_input + 1
    train_windows = int(total_windows * train_pers)
    test_windows = total_windows - train_windows

    train_start_index = first_index
    train_end_index = train_start_index + train_windows * n_input

    test_start_index = train_end_index
    test_end_index = test_start_index + test_windows * n_input

    train_data = df.iloc[train_start_index:train_end_index].to_numpy()
    test_data = df.iloc[test_start_index:test_end_index].to_numpy()

    if len(train_data) % n_input != 0:
        train_data = train_data[:-(len(train_data) % n_input)]
    if len(test_data) % n_input != 0:
        test_data = test_data[:-(len(test_data) % n_input)]
    train = array(split(train_data, len(train_data) // n_input))
    test = array(split(test_data, len(test_data) // n_input))

    return train, test

def to_supervised(train, n_input, n_out=7):
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        in_start += 1
    return np.array(X), np.array(y)
'''
def forecast(model, history, n_input):
    data = np.array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    input_x = data[-n_input:, :]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    yhat = model.model.predict(input_x, verbose=0)
    return yhat[0]
'''
def forecast(model, initial_data, steps):
    predictions = []
    current_data = initial_data.copy()

    for _ in range(steps):
        # Предсказываем следующий шаг
        next_step = model.predict(current_data)
        predictions.append(next_step[0, 0])

        # Обновляем данные для следующего прогноза
        current_data = np.roll(current_data, -1, axis=1)
        current_data[0, -1, :] = next_step

    return predictions
def evaluate_forecasts(actual, predicted):
    scores = list()
    for i in range(actual.shape[1]):
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        rmse = np.sqrt(mse)
        scores.append(rmse)
    s = 0

    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = np.sqrt(s / (actual.shape[0] * actual.shape[1]))

    day_mape = []
    for day in range(actual.shape[1]):
        mape = mean_absolute_percentage_error(actual[:, day], predicted[:, day]) * 100
        day_mape.append(mape)
        print(f'День {day + 1}: MAPE = {mape:.2f}%')

    return score, scores, day_mape

def evaluate_model_performance(model, train_x, train_y, test_x, test_y):
    test_pred = model.model.predict(test_x)

    test_pred = test_pred.numpy() if hasattr(test_pred, 'numpy') else test_pred
    test_y = test_y.numpy() if hasattr(test_y, 'numpy') else test_y

    rmse = np.sqrt(mean_squared_error(test_y, test_pred))
    mape = np.mean(np.abs((test_y - test_pred) / (test_y))) * 100

    return {
        'rmse': rmse,
        'mape': mape,
        'mape_eps_1e-8': np.mean(np.abs((test_y - test_pred) / (test_y + 1e-8))) * 100,
        'mape_eps_adaptive': np.mean(
            np.abs((test_y - test_pred) / (test_y + max(1e-8, 0.001 * np.max(test_y))))) * 100,
        'mae': np.mean(np.abs(test_y - test_pred)),
    }


import tempfile
import os


def model_to_bytes(model):
    """Сохраняет модель с указанием явной временной директории"""
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp_models')
    os.makedirs(temp_dir, exist_ok=True)

    temp_path = os.path.join(temp_dir, 'temp_model.keras')
    model.save(temp_path)

    with open(temp_path, 'rb') as f:
        model_bytes = f.read()

    os.remove(temp_path)  # Удаляем временный файл
    return model_bytes


def save_to_db(model, scaler, project_id, profile_name, n_steps, n_features, n_outputs, num_epoch, batch_size, slide_window, model_type, train_metrics, framework_version):
    scaler_bytes = io.BytesIO()
    pickle.dump(scaler, scaler_bytes)
    scaler_bytes.seek(0)

    db_model = Models.objects.create(
        project_id=project_id,
        profile_name=profile_name,
        num_epoch=num_epoch,
        batch_size=batch_size,
        slide_window=slide_window,
        name_neural='lstm',
        model_type=model_type,
        model_data=model_to_bytes(model),
        scaler_data=scaler_bytes.getvalue(),
        model_config={
            'n_steps': n_steps,
            'n_features': n_features,
            'n_outputs': n_outputs
        },
        created_at=timezone.now(),
        train_metrics=train_metrics,
        framework_version=framework_version
    )
    return db_model.id


import tempfile
import os
import shutil
import cloudpickle as cp

def load_from_db(model_id):
    db_model = Models.objects.get(id=model_id)

    # Создаем временную директорию для надежной работы с файлами
    temp_dir = tempfile.mkdtemp()
    try:
        # Сохраняем модель во временный файл
        model_path = os.path.join(temp_dir, f"model_{model_id}.keras")
        with open(model_path, 'wb') as f:
            f.write(db_model.model_data)

        # Загружаем модель
        model = tf.keras.models.load_model(model_path)

        # Загружаем скалер через joblib
        scaler_bytes = io.BytesIO(db_model.scaler_data)
        scaler = joblib.load(scaler_bytes)

        return model, scaler, db_model.model_config
    finally:
        # Всегда удаляем временную директорию
        shutil.rmtree(temp_dir, ignore_errors=True)