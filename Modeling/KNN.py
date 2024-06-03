import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 데이터 로드
data = pd.read_csv('C:/Users/82102/Desktop/전주시/혈당_혈압_최종.csv', encoding='cp949')

# 특성과 타겟 분리
X = data.drop('DIS', axis=1)
y = data['DIS']

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 적용 (3개의 주성분)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 모델 설정
knn = KNeighborsClassifier()

# 하이퍼파라미터 그리드 설정
param_grid = {
    'n_neighbors': [5, 19, 22],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# GridSearchCV 설정
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', verbose=1)

# 그리드 서치 실행
grid_search.fit(X_train, y_train)

# 최적의 파라미터와 최고 점수 출력
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# 테스트 데이터에 대한 예측과 평가
best_knn = grid_search.best_estimator_
y_pred_train = best_knn.predict(X_train)
y_pred_test = best_knn.predict(X_test)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

# 결과 출력
print("Train Accuracy: {:.2f}".format(train_accuracy))
print("Test Accuracy: {:.2f}".format(test_accuracy))