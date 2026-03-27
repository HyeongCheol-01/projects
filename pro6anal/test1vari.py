# 표준편차, 분산의 중요성

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

np.random.seed(42)

# 목표 평균
target_mean = 60
std_dev_small = 10
std_dev_large = 20

# data
class1_raw = np.random.normal(loc=target_mean, scale=std_dev_small, size = 100)
class2_raw = np.random.normal(loc=target_mean, scale=std_dev_large, size = 100)

print(class1_raw[:5])
# 평균 보정
class1_adj = class1_raw - np.mean(class1_raw) + target_mean
print(class1_adj[:5])
class2_adj = class2_raw - np.mean(class2_raw) + target_mean
print(class2_adj[:5])

# 정수화
# np.clip(배열, 최소값, 최대값)  np.clip(배열 10,60) -> (5, 45, 78, ...) => (10,45,60, ...)
class1 = np.clip(np.round(class1_adj), 10, 100).astype(int)
class2 = np.clip(np.round(class2_adj), 10, 100).astype(int)
print(class1)
print(class2)

# 통계 계산
mean1, mean2 = np.mean(class1), np.mean(class2)
std1, std2 = np.std(class1), np.std(class2)
var1, var2 = np.var(class1), np.var(class2)

print("1반(성적 편차 작음)")
print(f"평균:{mean1}, 표준편차:{std1}, 분산:{var1}")
print("2반(성적 편차 작음)")
print(f"평균:{mean2}, 표준편차:{std2}, 분산:{var2}")

df = pd.DataFrame({
    'class':['1반'] * 100 + ['2반'] * 100,
    'score':np.concatenate([class1, class2])
})
print(df.head())
df.to_csv('test1vari.csv', index=False, encoding='utf-8-sig')


print("시각화--------")
x1 = np.random.normal(1, 0.05, size=100)
x2 = np.random.normal(2, 0.05, size=100)

plt.figure(figsize=(10,6))
plt.scatter(x1, class1, alpha=0.8, label=f"1반(평균={mean1:2f}, 표준편차={std1:2f})")
plt.scatter(x2, class1, alpha=0.8, label=f"1반(평균={mean2:2f}, 표준편차={std2:2f})")
plt.hlines(target_mean, 0.5, 2.5, colors='red', \
        linestyles='dashed', label=f"공통평균={target_mean}")
plt.xticks([1,2], ['1반', '2반'])
plt.ylabel("시험 점수")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.boxplot([class1,class2], label=['1반','2반'])
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.hist(class1, bins=15, alpha=0.6, label='1반', edgecolor='black')
plt.hist(class2, bins=15, alpha=0.6, label='2반', edgecolor='blue')
plt.axvline(target_mean, color='red', \
    linestyle='dotted', label=f"공통평균={target_mean}")
plt.xlabel("시험 점수") 
plt.ylabel("빈도")
plt.legend()
plt.tight_layout()
plt.show()

# 국어 선생님 입장
# 귀무 가설(전통적 주장) : 두 반의 국어 점수의 표준편차(평균)는 차이가 없다
# 누군가가 실험을 통해 데이터 수집 후 두 반의 점수에 통계 계산 후 새로운 주장(의견)
# 대립 가설 : 두 반의 국어 점수의 표준편차(평균)는 차이가 있다.
# 가설검정(t-test)을 통해 두 의견의 채택, 기각을 판단할 수 있다.

# 가설검정은 모집단에 대한 어떤 가설을 설정하고 표본을 분석함으로써 그 가설의 타당성 여부를 검정하는 통계적 절차다.
# 표본추출된 자료를 통해 얻어진 값으로(검정 통계량) 모집단의 모수를 추측하는 작업.
# 모수의 상태에 대한 여러 주장들 중에서 어떤 주장을 사실로 받아들일지를 결정하는 과정
# 귀무가설(H0, 영가설) : 지금까지의 변함없는(일반적인) 생각.
# 대립가설(H1, 연구가설) : 귀무가설에 반하는 연구자의 의견
# 단계 :
# 1) 가설 수립 2) 통계량 계산 3) 가설선택 기준 수립 4) 판정 
# 







