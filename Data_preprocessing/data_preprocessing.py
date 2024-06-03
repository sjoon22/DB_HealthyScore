import pandas as pd

# 파일 경로
file_path = 'C:/Users/82102/Desktop/전주시/국민건강보험공단_건강검진정보_20221231 (1).CSV'

# CSV 파일 읽기
data = pd.read_csv(file_path, encoding='cp949')  # 인코딩 문제가 있다면 'cp949'로 변경

# 지정된 컬럼 제거
data = data.drop(columns=['구강검진수검여부', '치아우식증유무', '치석'])

# 결과를 새로운 CSV 파일로 저장
new_file_path = 'C:/Users/82102/Desktop/전주시/건강검진정보_attribute탈락.csv'
data.to_csv(new_file_path, index=False, encoding='cp949')  # 인코딩을 유지하고 싶다면 'cp949'를 지정