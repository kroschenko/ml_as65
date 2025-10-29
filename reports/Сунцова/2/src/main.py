import pandas as pd
from src.regression import run_regression
from src.classification import run_classification

def load_data():
    return pd.read_csv('data/winequality-white.csv', sep=';')

if __name__ == "__main__":
    df = load_data()
    run_regression(df)
    run_classification(df)
