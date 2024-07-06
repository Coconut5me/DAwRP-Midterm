#%% Import Lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

#%% Load data
# delim_whitespace=True để chỉ ịnh data được phân cách bằng khoảng trắng --> tìm các khoảng trắng và coi đó là dấu phân tách giữa các cột
df = pd.read_csv('Test/data/Ads.txt', delim_whitespace=True)
df = df.drop(columns=['Id'])

#%% Statistic
df_stat = df.describe()

#%% Pairplot
sns.pairplot(df)
plt.show()

#%% Heatmap
# Calculate the correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Heatmap of Correlation between Variables')
plt.show()

#%% Phân hoạch 3 mức
# Tạo các bins cho rời rạc hóa
bins = [0, 10, 20, np.inf]

# Nhãn cho các bins
labels = ['low', 'medium', 'high']

# Rời rạc hóa biến sales
df_sales_category = df.assign(sales_category=pd.cut(df['sales'], bins=bins, labels=labels, right=False))

# In ra 10 hàng đầu tiên của DataFrame để kiểm tra
print(df.head(10))

#%% Số lượng mẫu tin theo 3 mức phân hoạch
print("Số lượng mẫu tin theo 3 mức phân hoạch :")
print(df_sales_category['sales_category'].value_counts())

#%% Chia dữ liệu
X = df.drop(columns=['sales'])
y = df_sales_category['sales_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Fit model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#%% Test model
y_pred = model.predict(X_test)

# In ra các độ đo Precision, Recall, F1-Score
print(classification_report(y_test, y_pred))

#%% - Visualize Decision Tree
features = ['youtube','facebook','newspaper']
text_representation = tree.export_text(model, feature_names=features)
print(text_representation)

plt.figure(figsize=(20,20), dpi=150)
t = tree.plot_tree(model, feature_names=features, class_names=['low','medium','high'], filled=True)
plt.show()

#%% - Prediction
sales1=model.predict([[120,65,20]])
sales2=model.predict([[35,45,15]])


