import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv('bike_dataset.csv')

df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour

features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'hour']
X = df[features]
y = df['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

comparison = pd.DataFrame({'실제값': y_test, '예측값': np.round(y_pred, 0)})

print("예측값 / 실제값")
print(comparison)

print(f"\n결정계수: {r2:.4f}")

importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n예측 결과 및 변수 중요도")
print(importances)