#%% - Import Lib
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
from numpy import indices
from scipy import signal
from pmdarima import ARIMA
from scipy import signal
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from pmdarima.arima import auto_arima
from scipy import signal
from func import *
import warnings
warnings.filterwarnings("ignore")

#%% - Config
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 16

#%% Load data
df = pd.read_csv('Test/data/VNM.csv', index_col="Date", parse_dates=True)
df = df.iloc[::-1]
df.info()

#%% Covnvert Price type
df['Price'] = df['Price'].replace({',': ''}, regex=True).astype(float)
print(df.info())

#%% - Statistic
print(df['Price'].describe())

#%% - Draw chart
plt.plot(df['Price'])
plt.xlabel("Date")
plt.ylabel("Close prices")
plt.show()

#%% - After log
df_close = np.log(df['Price'])
plt.plot(df_close)
plt.xlabel('Date')
plt.ylabel('Price prices')
plt.show()

#%% - Statistic
print(df_close.describe())

#%% - Phân rã chuỗi dữ liệu
rolmean = df_close.rolling(12).mean()
rolstd = df_close.rolling(12).std()
plt.plot(df_close, color='blue', label='Original')
plt.plot(rolmean, color='red', label='Rolling mean')
plt.plot(rolstd, 'black', label='Rolling Std')
plt.legend()
plt.show()

# Phân rã chuỗi thời gian (decompose)
decompose_results = seasonal_decompose(df_close, model="multiplicative", period=30)
decompose_results.plot()
plt.show()

moving_avg = rolmean.dropna()

#%% - Retrieve outliers
def detect_outliers(series):
  """
    series: 1-D numpy array input
  """
  Q1 = np.quantile(series, 0.25)
  Q3 = np.quantile(series, 0.75)
  IQR = Q3-Q1
  lower_bound = Q1-1.5*IQR
  upper_bound = Q3+1.5*IQR
  lower_compare = series <= lower_bound
  upper_compare = series >= upper_bound
  outlier_idxs = np.where(lower_compare | upper_compare)[0]
  return outlier_idxs

outlier_idxs=detect_outliers(df_close)
print("Outlier indices: ", outlier_idxs)
print(len(outlier_idxs))
print("Outlier time: ", (df_close).index[outlier_idxs+1].values)
print("Outlier values: ", (df_close)[outlier_idxs])

#%% - Kiểm định tính dừng của dữ liệu (Station)
def adf_test(data):
    indices = ["ADF: Test statistic", "p-value", "# of Lags", "# of Observations"]
    test = adfuller(data, autolag="AIC")
    results = pd.Series(test[:4], index=indices)
    for key, value in test[4].items():
        results[f"Critical Value ({key})"] = value

    if results[1] <=0.05: # (p-value < 0.05)
        print("Rejected the null hypothesis (H0), \nthe data is non-stationary")
    else:
        print("Fail to reject the null hypothesis (H0), \nthe data non-stationary")

    return results

def kpss_test(data):
    indices = ["KPSS: Test statistic", "p-value", "# of Lags"]
    test = kpss(data)
    results = pd.Series(test[:3], index=indices)
    for key, value in test[3].items():
        results[f"Critical Value ({key})"] = value
    if results[1] <= 0.05:  # (p-value < 0.05)
        print("Rejected the null hypothesis (H0), \nthe data is stationary")
    else:
        print("Fail to reject the null hypothesis (H0), \nthe data is stationary")

    return results

print(adf_test(df_close))
print("-----"*5)
print(kpss_test(df_close))

#%% Kiểm định tự tương quan (Auto Correlation)
pd.plotting.lag_plot(df_close)
plt.show()

#%%
plot_pacf(df_close)
plt.show()

#%%
plot_acf(df_close)
plt.show()

#%% - Chuyển đổi dữ liệu --> chuỗi dừng
diff = df_close.diff(1).dropna()
#Biểu đồ thể hiên dữ liệu ban đầu và sau khi sai phân
fig, ax= plt.subplots(2, sharex="all")
df_close.plot(ax=ax[0], title="Gía đóng cửa")
diff.plot(ax=ax[1], title="Sai phân bậc nhất")
plt.show()

#%% - Boxplot
diff.plot(kind='box', title='Box Plot of Close')
plt.show()

#%% - Kiểm tra lại tính dừng của dữ liệu sau khi lấy sai phân
print(adf_test(diff))
print("-----"*5)
print(kpss_test(diff))
plot_pacf(diff) # --> xác định tham số "p" cho mô hình ARIMA
plt.show()

#%%
plot_acf(diff) # --> xác định tham số "q" cho mô hình ARIMA
plt.show()

#%% Divide data
train_data, test_data= train_test_split(df_close, test_size=0.2, shuffle=False)
plt.plot(train_data, 'blue', label = 'Train data')
plt.plot(test_data, 'red', label = 'Test data')
plt.xlabel('Date')
plt.ylabel('Close prices')
plt.legend()
plt.show()

#%% - Xác định tham số p,d,q cho mô hình ARIMA
stepwise_fit = auto_arima(train_data, trace=True, suppress_warnings=True)
print(stepwise_fit.summary())
stepwise_fit.plot_diagnostics(figsize=(15,8))
plt.show()

#%% - Tạo model
model = ARIMA(train_data, order=stepwise_fit.order, trend='t')
fitted = model.fit()
print(fitted.summary())

#%% - Dự báo (forecast)
fc = fitted.get_forecast(len(test_data))
fc_values = fc.predicted_mean
fc_values.index = test_data.index
conf = fc.conf_int(alpha=0.05) #95% conf
lower_series = conf['lower Price']
lower_series.index = test_data.index
upper_series = conf['upper Price']
upper_series.index = test_data

#%% - Đánh gía hiệu suất mô hình
mse = mean_squared_error(test_data, fc_values)
print('Test MSE: %3f' % mse)
rmse = math.sqrt(mse)
print('Test RMSE: %3f' % rmse)

#%% - Calculate RMSE for baseline
baseline_prediction = np.full_like(test_data, train_data.mean()) #median
baseline_rmse = np.sqrt(mean_squared_error(test_data, baseline_prediction))

#%% - Plot actual vs predicted values
plt.figure(figsize=(16, 10), dpi=150)
plt.plot(train_data, label="Training data")
plt.plot(test_data, color='orange', label="Actual stock price")
plt.plot(fc_values, color='red', label="Predicted stock price")  # Include predicted values in the same plot
plt.fill_between(lower_series.index, lower_series, upper_series, color='b', alpha=.10)
plt.title("Stock price prediction")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.legend(loc='upper left')
plt.show()

#%% - Visualize RMSE comparison
print('ARIMA Model RMSE: {:.2f}'.format(rmse))
print('Baseline RMSE: {:2f}'.format(baseline_rmse))

plt.figure(figsize=(16,10), dpi=150)
plt.bar(['ARIMA Model', 'Baseline'], [rmse, baseline_rmse], color=['blue','green'])
plt.title('Root Mean Squared Error (RMSE) Comparison')
plt.ylabel("RMSE")
plt.show()

#%% Split data
train_data, test_data= train_test_split(df_close, test_size=0.1, shuffle=False)
train_list_expanding, val_list_expanding, df_fold_expanding_window = expanding_window(df_close, 5, 0.8)

#%% Evaluation
columns = ['Model', 'MAE_val', 'MSE_val', 'RMSE_val', 'MAE_test', 'MSE_test', 'RMSE_test']
df_evaluation = pd.DataFrame(columns=columns)

#%%
for i in range(len(train_list_expanding)):
        stepwise_fit = auto_arima(train_data, trace=True, suppress_warnings=True)
        model = ARIMA(train_data, order=stepwise_fit.order, trend='t')
        fitted = model.fit()

        # - Forecasting
        fc = fitted.get_forecast(len(val_list_expanding[i]))
        fc_values = fc.predicted_mean
        fc_values.index = val_list_expanding[i].index

        fc_test = fitted.get_forecast(len(test_data))
        fc_values_test = fc_test.predicted_mean
        fc_values_test.index = test_data.index

        # - Confidence interval
        conf_int = fc.conf_int()

        # - Evaluation metrics
        mae_val = mean_absolute_error(val_list_expanding[i], fc_values)
        mse_val = mean_squared_error(val_list_expanding[i], fc_values)
        rmse_val = math.sqrt(mse_val)

        mae_test = mean_absolute_error(test_data, fc_values_test)
        mse_test = mean_squared_error(test_data, fc_values_test)
        rmse_test = math.sqrt(mse_test)

        # - Calculate RMSE for baseline
        baseline_prediction = np.full_like(val_list_expanding[i], train_list_expanding[i].mean())  # median
        baseline_rmse = np.sqrt(mean_squared_error(val_list_expanding[i], baseline_prediction))

        # Append evaluation metrics to the DataFrame
        df_evaluation = df_evaluation._append({'Model': f'Fold_{i}',
                                               'MSE_val': mse_val,
                                               'RMSE_val': rmse_val,
                                               'MAE_val': mae_val,
                                               'MSE_test': mse_test,
                                               'RMSE_test': rmse_test,
                                               'MAE_test': mae_test,
                                               }, ignore_index=True)
        # - Plot actual vs predicted values
        plt.figure(figsize=(16, 10), dpi=150)
        plt.plot(df_close)
        plt.plot(train_list_expanding[i], label="Training data")
        plt.plot(val_list_expanding[i], color='orange', label="Actual stock price")
        plt.plot(fc_values, color='red', label="Predicted stock price")  # Include predicted values in the same plot
        plt.title("Stock price prediction")
        plt.xlabel("Time")
        plt.ylabel("Stock price")
        plt.legend(loc='upper left')
        plt.show()



        # - Visualize RMSE comparison
        print('ARIMA Model RMSE: {:.2f}'.format(rmse_val))
        print('Baseline RMSE: {:2f}'.format(baseline_rmse))

        plt.figure(figsize=(16, 10), dpi=150)
        plt.bar(['ARIMA Model', 'Baseline'], [rmse_val, baseline_rmse], color=['blue', 'green'])
        plt.title('Root Mean Squared Error (RMSE) Comparison')
        plt.ylabel("RMSE")
        plt.show()