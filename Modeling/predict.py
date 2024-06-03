import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib

# 원본 데이터셋을 불러와 훈련된 모델 설정을 복구
data_original = pd.read_csv('C:/Users/82102/Desktop/전주시/혈당_혈압_최종.csv', encoding='cp949')

# 식전혈당(공복혈당)을 제외한 열 선택
X_original = data_original.drop(columns=['DIS', '식전혈당(공복혈당)'])
y_original = data_original['DIS']

# 원본 데이터에 대한 스케일링 및 PCA 변환
scaler = StandardScaler()
scaler.fit(X_original)  # 원본 데이터에 스케일러 적용

pca = PCA(n_components=3)
pca.fit(scaler.transform(X_original))  # 스케일링된 원본 데이터에 PCA 적용

# 최적의 파라미터를 사용하여 KNN 모델 설정
knn = KNeighborsClassifier(metric='manhattan', n_neighbors=22, weights='uniform')
knn.fit(pca.transform(scaler.transform(X_original)), y_original)  # 전체 원본 데이터셋에 모델 훈련

# 모델 저장
joblib.dump((scaler, pca, knn), 'health_model.pkl')

print("모델이 저장되었습니다.")