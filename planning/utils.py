import pandas as pd

def process_excel(file):
    df = pd.read_excel(file)
    return df