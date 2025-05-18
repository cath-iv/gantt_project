import pandas as pd
import requests

def process_excel(file):
    """Обработка Excel-файла"""
    df = pd.read_excel(file)
    # Ваша предобработка данных
    return df

def get_weather(date):
    """Заглушка для погоды - можно подключить реальный API позже"""
    return {
        'temp': 20,
        'wind': 5,
        'rain': 0,
        'description': 'ясно'
    }