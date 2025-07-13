import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data_path = r'C:\Users\游晨仪\Desktop\w2\US-pumpkins.csv'
pumpkin_data = pd.read_csv(data_path)


pumpkin_data = pumpkin_data.dropna(subset=['Low Price', 'High Price', 'Item Size', 'Origin', 'Variety', 'City Name'])

pumpkin_data['Price Range'] = pumpkin_data['High Price'] - pumpkin_data['Low Price']

features = ['Item Size', 'Origin', 'Variety', 'City Name', 'Package']
target = 'Low Price'


X = pumpkin_data[features]
y = pumpkin_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numeric_features = []
categorical_features = features  # 所有特征都是分类的

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


linear_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])


linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("\n线性回归模型评估结果:")
print(f"均方误差(MSE): {mse_linear}")
print(f"R平方值(R2 Score): {r2_linear}")


rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n随机森林模型评估结果:")
print(f"均方误差(MSE): {mse_rf}")
print(f"R平方值(R2 Score): {r2_rf}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, alpha=0.5)
plt.xlabel('实际最低价格')
plt.ylabel('预测最低价格')
plt.title('线性回归: 实际 vs 预测')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.xlabel('实际最低价格')
plt.ylabel('预测最低价格')
plt.title('随机森林: 实际 vs 预测')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')

plt.tight_layout()
plt.show()

ohe = linear_model.named_steps['preprocessor'].named_transformers_['cat']
feature_names = ohe.get_feature_names_out(input_features=features)


importances = rf_model.named_steps['regressor'].feature_importances_

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values('Importance', ascending=False)

print("\n随机森林模型特征重要性(前10):")
print(importance_df.head(10))