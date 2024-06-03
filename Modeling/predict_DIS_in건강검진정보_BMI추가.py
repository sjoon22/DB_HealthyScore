import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# 원본 데이터셋을 불러와 훈련된 모델 설정을 복구
data_original = pd.read_csv('C:/Users/82102/Desktop/전주시/혈당_혈압_최종.csv', encoding='cp949')
X_original = data_original.drop('DIS', axis=1)
y_original = data_original['DIS']

# 원본 데이터에 대한 스케일링 및 PCA 변환
scaler = StandardScaler()
scaler.fit(X_original)  # 원본 데이터에 스케일러 적용

pca = PCA(n_components=3)
pca.fit(scaler.transform(X_original))  # 스케일링된 원본 데이터에 PCA 적용

# 새 데이터셋 불러오기
new_data = pd.read_csv('C:/Users/82102/Desktop/전주시/건강검진정보_BMI추가.csv', encoding='cp949')

# 새 데이터셋이 'DIS'를 제외한 필요한 모든 열을 포함하고 있다고 가정
X_new = new_data[X_original.columns]

# 같은 스케일링 및 PCA 변환 적용
X_new_scaled = scaler.transform(X_new)
X_new_pca = pca.transform(X_new_scaled)

# GridSearch에서 찾은 최적의 파라미터를 사용하여 모델 로드
model = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs')
model.fit(pca.transform(scaler.transform(X_original)), y_original)  # 전체 원본 데이터셋에 모델 훈련

# 모델을 사용한 예측
new_data['DIS_predicted'] = model.predict(X_new_pca)

# 선택적으로 예측된 'DIS' 값이 포함된 새 데이터셋을 저장
new_data.to_csv('C:/Users/82102/Desktop/전주시/건강검진정보_DIS예측까지.csv', index=False, encoding='cp949')

print("예측 결과가 데이터셋에 추가되어 저장되었습니다.")