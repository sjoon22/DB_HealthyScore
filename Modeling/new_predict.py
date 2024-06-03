import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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

# 모델 파이프라인 구축
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, C=0.1, penalty='l2', solver='lbfgs'))
])

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 훈련
model_pipeline.fit(X_train, y_train)

# 훈련 및 테스트 결과 예측
y_train_pred = model_pipeline.predict(X_train)
y_test_pred = model_pipeline.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# 결과 출력
print("Train Accuracy: {:.2f}".format(train_accuracy))
print("Test Accuracy: {:.2f}".format(test_accuracy))

# 훈련된 모델에서 특성의 가중치 출력
feature_importance = model_pipeline.named_steps['classifier'].coef_.flatten()

# 인코딩된 특성 이름 얻기
encoded_feature_names = (
    numeric_features + 
    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
)

# 각 특성의 이름과 가중치를 데이터프레임으로 정리
feature_importance_df = pd.DataFrame({
    'Feature': encoded_feature_names,
    'Weight': feature_importance
})

# 가중치 절대값으로 정렬
feature_importance_df['Absolute_Weight'] = feature_importance_df['Weight'].abs()
feature_importance_df = feature_importance_df.sort_values(by='Absolute_Weight', ascending=False)

# 가중치 데이터프레임 저장
feature_importance_df[['Feature', 'Weight']].to_csv('C:/Users/82102/Desktop/전주시/컬럼별가중치_2.csv', index=False, encoding='cp949')

print("가중치 데이터프레임이 저장되었습니다.")