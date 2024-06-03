import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

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

# 모델 파이프라인 구축
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, C=0.1, penalty='l2', solver='lbfgs'))
])

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 훈련
model_pipeline.fit(X_train, y_train)

# 테스트 세트의 예측 확률
probs = model_pipeline.predict_proba(X_test)[:, 1]  # 클래스 1의 확률
probs_rounded = [round(prob, 2) for prob in probs]  # 확률을 소수점 둘째 자리까지 반올림

# 테스트 세트의 예측 결과에 확률 추가
X_test_reset = X_test.reset_index(drop=True)
test_results = pd.DataFrame(X_test_reset)
test_results['label_1_probability'] = probs_rounded

# 결과 저장
test_results.to_csv('C:/Users/82102/Desktop/전주시/확률까지들어간파일.csv', index=False, encoding='cp949')

print("파일에 확률이 추가되어 저장되었습니다.")