#%% - Import Lib
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import warnings
warnings.filterwarnings("ignore")

#%% - Config
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 16

#%% - Load data
df = pd.read_excel('Test/data/Data.xls', index_col="Date", parse_dates=True)
df.info()

#%% Calculate
mean_value = np.mean(df['Value'])

#%% Draw chart
# Vẽ đồ thị dữ liệu và đường trung bình
plt.plot(df['Value'], label='Data')
plt.axhline(y=mean_value, color='r', linestyle='--', label='Mean Value')
plt.text(df.index[-1], mean_value, f'Mean: {mean_value:.2f}', ha='right', va='center')

plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

#%% Vẽ biểu đồ histogram của chuỗi dữ liệu
plt.hist(df['Value'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Time Series Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

