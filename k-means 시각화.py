import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. CSV 파일 불러오기 (cp949 인코딩)
input_csv_path = 'final_data_with_kmeans.csv'
df = pd.read_csv(input_csv_path, encoding='cp949')

# 2. 분석에 사용할 텍스트 컬럼 지정 (n_gram 컬럼 사용)
n_gram_column = 'n_gram'
if n_gram_column not in df.columns:
    raise ValueError(f"'{n_gram_column}' 컬럼이 존재하지 않습니다. 컬럼명을 확인하세요.")

# 3. TF-IDF 벡터화 수행 (n_gram 텍스트)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df[n_gram_column].astype(str))

# 4. (선택) KMeans 재적용 (이미 CSV에 cluster_label이 있다면 생략 가능)
#   - 만약 CSV에 이미 cluster_label이 있고, 그대로 쓰고 싶다면 주석 처리하세요.
k = 3  # 클러스터 수 (데이터에 맞게 조정)
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster_label'] = kmeans.fit_predict(X)  # 다시 클러스터링

# 5. PCA로 2차원 축소
pca = PCA(n_components=2, random_state=42)
X_dense = np.asarray(X.todense())
X_pca = pca.fit_transform(X_dense)

df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

# 6. 클러스터 중심점(centroid)도 PCA 변환하여 표시
#    (위에서 KMeans를 재적용했다면, kmeans.cluster_centers_ 사용 가능)
centroids = kmeans.cluster_centers_        # 원본 TF-IDF 공간에서의 중심점
centroids_pca = pca.transform(centroids)   # PCA 2차원 공간으로 변환

# 7. 산점도 시각화
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
clusters = sorted(df['cluster_label'].unique())

for cluster in clusters:
    subset = df[df['cluster_label'] == cluster]
    plt.scatter(subset['pca1'], subset['pca2'],
                label=f'Cluster {cluster}',
                color=colors[int(cluster) % len(colors)],
                alpha=0.7)

# 클러스터 중심점 표시 (검은색 X 마커)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            marker='X', s=200, c='black', label='Centroids')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('KMeans Clustering Visualization with Centroids')
plt.legend()
plt.grid(True)
plt.show()
