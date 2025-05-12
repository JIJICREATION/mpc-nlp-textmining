import os
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. 입력 폴더와 출력 폴더 지정
input_folder = r'C:\Users\abwm2\Desktop\BootCamp\TIL\JIMINSEONG\jiminseong\극성분석\tf idf'  # TF-IDF CSV 파일들이 들어있는 폴더 경로로 수정하세요.
output_folder = os.path.join(input_folder, 'unsupervised_results')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"출력 폴더 생성됨: {output_folder}")
else:
    print(f"출력 폴더 존재: {output_folder}")

# 2. 입력 폴더 내 CSV 파일 목록 가져오기
csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
print(f"총 {len(csv_files)}개의 CSV 파일 발견됨.")

# 3. 각 파일별로 비지도 극성 분류 수행
for file in csv_files:
    print(f"\n===== 파일 처리 시작: {file} =====")
    try:
        # 파일 읽기 (인코딩은 상황에 맞게 'utf-8-sig' 또는 'cp949'로 수정)
        df = pd.read_csv(file, encoding='utf-8-sig')
    except Exception as e:
        print(f"파일 로드 실패: {file}, 에러: {e}")
        continue

    # 4. TF-IDF 피처만 사용 (날짜 등 다른 컬럼이 있다면 제외)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        print("수치형 피처가 없습니다. 파일을 건너뜁니다.")
        continue
    print("TF-IDF 피처 크기:", numeric_df.shape)

    # 5. 문서가 2개 이상인 경우에만 KMeans 클러스터링 수행, 문서 1개인 경우 강제로 클러스터 할당
    if numeric_df.shape[0] > 1:  # 문서가 2개 이상인 경우
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(numeric_df)
    else:  # 문서가 1개인 경우 강제로 클러스터 할당
        clusters = np.array([0])  # 강제로 군집 0에 할당
        print("문서가 1개이므로 군집 0에 할당됨.")

    # 6. 클러스터 결과를 원본 데이터에 추가
    df['cluster'] = clusters
    print("클러스터 분포:")
    print(df['cluster'].value_counts())

    # 7. PCA 시각화 (TF-IDF 피처가 2개 이상일 때만 수행)
    if numeric_df.shape[1] >= 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(numeric_df)
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.title(f"KMeans Clustering for {os.path.basename(file)}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(label="Cluster")
        plt.show()
    else:
        print("TF-IDF 피처가 1개이므로 PCA 시각화를 건너뜁니다.")

    # 8. 결과 파일 저장: 원본 파일명에 '_clustered' 접미사 추가하여 출력 폴더에 저장
    base_filename = os.path.splitext(os.path.basename(file))[0]
    output_file = os.path.join(output_folder, base_filename + '_clustered.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"결과 파일 저장됨: {output_file}")

print("모든 파일 처리 완료.")
