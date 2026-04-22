






import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

#샘플 데이터 생성 
x, y = make_moons(n_samples=200, noise=0.05, shuffle=True, random_state=0)
print(x[:5], x.shape)   # (200, 2)
#print(y)

plt.scatter(x[:,0], x[:,1])
plt.show()

print('KMeans로 군집 분류')
km= KMeans(n_clusters=2, init='k-means++', random_state=0)
pred1= km.fit_predict(x)
print('km 예측 군집 id :', pred1[:10])

#km 결과 시각화
def plotResult(x,pr):
    plt.scatter( x[pr==0, 0], x[pr==0, 1], c='blue', marker='o', s=40, label='cluster1')
    plt.scatter( x[pr==1, 0], x[pr==1, 1], c='red', marker='s', s=40, label='cluster2')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='black', marker='+', s=40, label='centroid')
    plt.title('Clutering Result')
    plt.legend()
    plt.show()
    
plotResult(x,pred1)

print('DBSCAN으로 군집 분류')
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
# eps=0.2 - 샘플간 최대거리, min_samples=5 - 점에 대한 이웃 샘플수
pred2 = db.fit_predict(x)
print('DBSCAN 군집 id :', pred2[:10])
print('군집 종류 :', set(pred2))    # 0, 1 이상치는 없는 상태

plotResult(x,pred2)     # 분류가 제대로 됨. 모양따라 군집형성
# KMeans는 k개에 군집의 갯수를 맞추고, DBSCAN은 밀도의 형태를 맞춘다.




