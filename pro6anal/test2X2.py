# 교차분석(카이제곱, 카이스퀘어) 가설 검정
# 범주형 자료(변수)를 대상으로 교차빈도에 대한 기술통계량을 제공
# 교차 빈도에 대한 통계적 유의성을 검증해 주는 분석기법
# 분산이 퍼져 있는 모습을 분포로 만든 것이 카이제곱 분포
# X² = ∑(관측값 - 기대값)² / 기대값

# 적합도 : 일원카이제곱 검정 
# 독립성, 동질성 검정

# 가설을 채택하는 두 가지 방법 연습------

import pandas as pd
data = pd.read_csv("pass_cross.csv", encoding="euc-kr")
print(data.head())
print(data.shape)   # 50,4
print(data.shape[0])
print(data.shape[1])
print()


# 귀무가설 : 벼락치기 공부하는 것과 합격여부는 관계가 없다.
print(data[(data['공부함'] == 1) & (data['합격']==1)].shape[0])     # 18
print(data[(data['공부함'] == 1) & (data['불합격']==1)].shape[0])   # 7
print('빈도표 작성 -------------')
ctab = pd.crosstab(index=data['공부안함'], columns=data['불합격'], margins=True)
ctab.columns = ['합격', '불합격', '행합']
ctab.index = ['공부함', '공부안함', '열합']
print(ctab) # 관찰값(빈도)

# 검정방법1 : 카이제곱표
# 기대도수 = (각행의 주변합) * (각열의 주변합) / 총합<전체표본수>

print((18-15)** 2 / 15 + (7-10)** 2 / 10 + (12-15)**2 / 15 + (13-10)*2 / 10)
# chi2 = 3.0
# df = 2 - 1 = 1
# 유의수준 : 0.05
# 임계값 : 3.84
# 판정 : 카이제곱 검정통계량:3 < 임계값(3.84) 이므로 귀무 채택역 내에 있으므로 귀무가설 채택!
# 그러므로 벼락치기 공부하는 것과 합격여부는 관계가 없다. 는 의견 유지

# 검정밥법2 : p-value 사용
import scipy.stats as stats
chi2, p, _, _ = stats.chi2_contingency(ctab)
print(chi2, p) # chi2:3.0  p:0.5578254003710748
# 판정 : 유의수준 0.05 < p:0.557825 이므로 귀무 채택
# 검정에 사용된 자료를 우연히 발생한 자료라고 할 수 있다.

# chi2 검정 정식명칭 : Perason's chi-suare test
# 두 개의 불 연속변수(범주형) 간의 상관관계를 측정. 관찰빈도가 통계적으로 유의한 지 확인








