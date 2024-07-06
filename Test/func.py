import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
from statsmodels.tsa.stattools import adfuller, kpss
from pmdarima import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import warnings
warnings.filterwarnings("ignore")


#%%
# Expanding window
def expanding_window(df_train, n_folds, split_train_rate):
    df_fold_expanding_window = pd.DataFrame(columns=['Fold', 'start_index', 'train_length', 'val_length', 'end_index'])

    train_list=[]
    val_list=[]
    start_index = 0
    window_size = len(df_train) // n_folds
    end_index = window_size
    first_window_length = int(window_size * (1-split_train_rate))

    for i in range(n_folds):
        # Extract data within the window
        window_data = df_train[:end_index]

        # Split the window data into training and val sets
        val_data = window_data[-first_window_length:]
        train_data = window_data[:-len(val_data)]
        train_list.append(train_data)
        val_list.append(val_data)

        # Create a Series containing information about the current fold
        fold_info = pd.Series({
            'Fold': f'Fold {i + 1}',
            'start_index': start_index,
            'train_length': len(train_data),
            'val_length': len(val_data),
            'end_index': end_index
        })

        # Add fold information to the DataFrame
        df_fold_expanding_window = df_fold_expanding_window._append(fold_info, ignore_index=True)

        # Increment the indices for the next window
        start_index = 0
        end_index += window_size

    return train_list, val_list, df_fold_expanding_window



