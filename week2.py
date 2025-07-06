# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# 数据加载
data_path = r'C:\Users\游晨仪\Desktop\w2\US-pumpkins.csv'
pumpkin_data = pd.read_csv(data_path)



# 数据预处理
# 查看数据前几行和基本信息
print("数据前五行:\n", pumpkin_data.head())
print("\n数据基本信息:\n", pumpkin_data.info())

# 处理缺失值
print("\n缺失值统计:\n", pumpkin_data.isnull().sum())
pumpkin_data = pumpkin_data.dropna(subset=['Low Price', 'High Price', 'Item Size', 'Origin'])

# 特征工程
# 选择可能影响价格的特征
features = ['Item Size', 'Origin']  # 分类特征需要编码
target = 'Low Price'

# 对分类特征进行标签编码
label_encoders = {}
for col in features:
    if pumpkin_data[col].dtype == 'object':
        le = LabelEncoder()
        pumpkin_data[col] = le.fit_transform(pumpkin_data[col].astype(str))
        label_encoders[col] = le

# 准备数据
X = pumpkin_data[features]
y = pumpkin_data[target]

#  数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
model = LinearRegression()
model.fit(X_train, y_train)

#  模型评估
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n模型评估结果:")
print(f"均方误差(MSE): {mse}")
print(f"R平方值(R2 Score): {r2}")

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('实际最低价格')
plt.ylabel('预测最低价格')
plt.title('实际价格 vs 预测价格')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
plt.show()

# 查看模型系数
print("\n模型系数分析:")
for i, col in enumerate(features):
    print(f"{col}的系数: {model.coef_[i]}")
print("截距:", model.intercept_)
