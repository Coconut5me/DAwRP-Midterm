#%% Import Lib
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle
import warnings
warnings.filterwarnings("ignore")

#%% Load data
file_path = "Test/data/Spending_data.json"
with open(file_path, "r") as f:
    data = json.load(f)
df = pd.DataFrame(data)

#%% Headmap
df_numeric = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Heatmap of Correlation between Variables')
plt.show()

#%%
X = df['Income (VND)']
y = df['Expenditure (VND)']

#%% - Create model
df['intercept'] = 1

model = sm.OLS(y, X).fit()
print(model.summary())

#%% - Save model
with open('Test/models/linearRegression_c3', 'wb') as f:
    pickle.dump(model, f)

#%% Load the saved model
with open('Test/models/linearRegression_c3', 'rb') as f:
    saved_model = pickle.load(f)

#%% - Prediction
income_value = 26
income_value_with_intercept = [1, income_value]
pred_value = saved_model.predict(income_value_with_intercept)
print("Predicted expenditure with income =", income_value, "is", pred_value[0])
