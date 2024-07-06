#%% Import Lib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xml.etree.ElementTree as ET
import statsmodels.api as sm
import pickle
from sklearn.linear_model import LinearRegression

#%% Load data
tree = ET.parse('Test/data/XeDaQuaSuDung.xml')
root = tree.getroot()

data = []

for xe in root.findall('xe'):
    hang_sx = xe.find('Hang_SX').text
    so_km = float(xe.find('So_Km').text)
    nam_su_dung = int(xe.find('Nam_Su_Dung').text)
    gia_xe = float(xe.find('Gia_Xe').text)

    data.append({'hang_sx': hang_sx, 'so_km': so_km, 'nam_su_dung': nam_su_dung, 'gia_xe': gia_xe})

df = pd.DataFrame(data)

print(df.describe())

#%% Biểu đồ phân tán
# Biểu đồ phân tán của giá xe theo số năm sử dụng
plt.figure(figsize=(10, 6))
plt.scatter(df['nam_su_dung'], df['gia_xe'], color='blue', alpha=0.5)
plt.title('Phân tán giá xe theo số năm sử dụng')
plt.xlabel('Số năm sử dụng')
plt.ylabel('Giá xe')
plt.show()

# Biểu đồ phân tán của giá xe theo số km đã đi
plt.figure(figsize=(10, 6))
plt.scatter(df['so_km'], df['gia_xe'], color='green', alpha=0.5)
plt.title('Phân tán giá xe theo số km đã đi')
plt.xlabel('Số km đã đi')
plt.show()

#%% Vẽ heatmap
df_drop_brand = df.drop(columns=['hang_sx'])

# Calculate the correlation matrix
corr_matrix = df_drop_brand.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Heatmap of Correlation between Variables')
plt.show()


#%% Tính toán VIF
def calculate_vif(df):
    features = list(df.columns)
    num_features = len(features)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [sm.OLS(df[feature], sm.add_constant(df.drop(columns=[feature]))).fit().rsquared for feature in
                       features]

    return vif_data


# In kết quả VIF
vif_result = calculate_vif(df_drop_brand)
print(vif_result)

#%% Create model
df['intercept'] = 1

# Define independent variables (features) and dependent variable (target)
X = df[['nam_su_dung', 'so_km', 'intercept']]
y = df['gia_xe']

#%% Fit model
model = sm.OLS(y, X).fit()
print(model.summary())

#%% - Save model
with open('Test/models/linearRegression_c2', 'wb') as f:
    pickle.dump(model, f)

#%% Load the saved model
with open('Test/models/linearRegression_c2', 'rb') as f:
    saved_model = pickle.load(f)

#%% Tạo dữ liệu mới để dự đoán
new_data = {'nam_su_dung': [6, 3], 'so_km': [112, 165], 'intercept': [1, 1]}

predicted_prices = []

for i in range(len(new_data['nam_su_dung'])):
    nam_su_dung = new_data['nam_su_dung'][i]
    so_km = new_data['so_km'][i]

    prediction = saved_model.predict([nam_su_dung, so_km, 1])
    predicted_prices.append(round(prediction[0], 2))

df_predicted = pd.DataFrame(new_data)
df_predicted['predicted_price'] = predicted_prices
df_predicted = df_predicted.drop(columns=['intercept'])

