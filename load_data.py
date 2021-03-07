import numpy as np
import pandas as pd


def load_data(file_X, file_y):
    X_df = pd.read_csv(file_X).drop(columns='Unnamed: 0')
    y_df = pd.read_csv(file_y).drop(columns='Unnamed: 0')
    return X_df.to_numpy().astype(float), y_df.to_numpy().astype(float)
