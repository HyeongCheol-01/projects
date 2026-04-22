# 어느 쇼핑몰의 고객 행동 데이터를 이용해 군집 분류


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
#일반적으로 계층적/비계층적 군집 분석을 선행하고, 마음에 안들면 DBSCAN을 함

np.random.seed(42)

#data 생성
# vip 고객
vip = pd.DataFrame({
    'annual_spending':np.random.normal(700,40,80),
    'visit_per_month':np.random.normal(20,2,80),
    'avg_purchase':np.random.normal(80,10,80),
    'group':'vip'
})

#일반고객
normal = pd.DataFrame({
    'annual_spending':np.random.normal(300,100,150),
    'visit_per_month':np.random.normal(10,4,150),
    'avg_purchase':np.random.normal(30,15,150),
    'group':'normal'
})
    
#저활동 고객 - 방문 적음, 구매 적음
low = pd.DataFrame({
    'annual_spending':np.random.normal(100,30,70),
    'visit_per_month':np.random.normal(3,1,70),
    'avg_purchase':np.random.normal(10,5,70),
    'group':'low'
})
print(low.head(2))

# 특이 패턴 고객 (비선형 패턴) - 일정하지 않은 소비 패턴
t=np.linspace(0, 3*np.pi, 60)
curve = pd.DataFrame({
    'annual_spending':np.random.normal(0,10,len(t)) + 200 + 100*np.cos(t),
    'visit_per_month':np.random.normal(0,1,len(t)) + 10 + 5 *np.sin(t),
    'avg_purchase': 40 + 10 * np.sin(t),
    'group':'curve'
})

#이상 고객 (이상치) - 너무 많이 사거나 거의 안사거나
outliers = pd.DataFrame({
    'annual_spending':[900,50,850],
    'visit_per_month':[10,1,25],
    'avg_purchase':[120,5,100],
    'group':'outliers'
})

#data 합치기
df = pd.concat([vip,normal,low,curve,outliers], ignore_index=True)
print(df.head(2))

print()
# 초기 데이터 시각화
plt.figure(figsize=(6,5))
sns.scatterplot(
    x=df['annual_spending'],
    y=df['visit_per_month'],
    hue=df['group'],
    palette='Set2'
)
plt.title('원본 데이터')
plt.xlabel('연간 소비액')
plt.ylabel('월 방문 수')
plt.legend(title='소비 행태')
plt.show()

#DBSCAN
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df.drop(columns=['group']))
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
clusters = dbscan.fit_predict(x_scaled)
df['cluster'] = clusters


# 군집 결과 시각화
plt.figure(figsize=(6,5))
sns.scatterplot(
    x=df['annual_spending'],
    y=df['visit_per_month'],
    hue=df['cluster'],
    palette='Set1'
)
plt.title('군집 결과')
plt.xlabel('연간 소비액')
plt.ylabel('월 방문 수')
plt.legend(title='소비 행태')
plt.show()  # 매출+방문+구매금액에 따라 3개의 군집으로 분류함

print()
print('각 군집 평균:', \
    df.groupby('cluster')[['annual_spending', 'visit_per_month','avg_purchase']].mean())
