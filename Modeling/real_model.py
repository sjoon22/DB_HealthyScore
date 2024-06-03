import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

# 데이터 로드
data = pd.read_csv('C:/Users/82102/Desktop/전주시/건강검진정보_진짜모델돌릴데이터.csv', encoding='cp949')

# 사용하지 않는 '시도코드' 컬럼 제거
data.drop(columns=['시도코드'], inplace=True)

# 숫자형 특성과 범주형 특성 정의
numeric_features = [
    '연령대코드(5세단위)', '허리둘레', '시력(좌)', '시력(우)', '수축기혈압', '이완기혈압', '식전혈당(공복혈당)',
    '총콜레스테롤', '트리글리세라이드', 'HDL콜레스테롤', 'LDL콜레스테롤', '혈색소', '혈청크레아티닌',
    '혈청지오티(AST)', '혈청지피티(ALT)', '감마지티피', 'BMI'
]
categorical_features = ['성별', '요단백', '흡연상태', '음주여부']

# 데이터 전처리를 위한 파이프라인 설정
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# 데이터와 타겟 분리
X = data.drop('DIS_predicted', axis=1)
y = data['DIS_predicted']

# 전처리와 로지스틱 회귀를 결합한 파이프라인 생성
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, C=0.1, penalty='l2', solver='lbfgs'))
])

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 보정된 분류기를 사용해 모델 파이프라인 훈련 및 보정
calibrated_clf = CalibratedClassifierCV(estimator=model_pipeline, method='sigmoid', cv=5)
calibrated_clf.fit(X_train, y_train)

# 예측 및 확률 계산
y_pred = calibrated_clf.predict(X_test)
probs = calibrated_clf.predict_proba(X_test)[:, 1]
probs_rounded = [round(prob, 2) for prob in probs]  # 확률을 소수점 둘째 자리까지 반올림

# 결과 저장
test_results = pd.DataFrame(X_test).reset_index(drop=True)
test_results['Predicted_label'] = y_pred
test_results['label_1_probability'] = probs_rounded
test_results.to_csv('C:/Users/82102/Desktop/전주시/확률까지들어간파일.csv', index=False, encoding='cp949')

# 모델 성능 평가
train_accuracy = accuracy_score(y_train, calibrated_clf.predict(X_train))
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")