import pandas as pd

# 수정된 파일 경로
file_path = 'C:/Users/82102/Desktop/전주시/건강검진정보_attribute탈락.csv'

# CSV 파일 읽기
data = pd.read_csv(file_path, encoding='cp949')  # 파일 인코딩에 맞게 조정

# NaN 값이 포함된 행 제거
data_cleaned = data.dropna()

# 결과를 새로운 CSV 파일로 저장
new_file_path = 'C:/Users/82102/Desktop/전주시/건강검진정보_null제거.csv'
data_cleaned.to_csv(new_file_path, index=False, encoding='cp949')

# 데이터 행 개수 출력
print("데이터 행 개수:", data_cleaned.shape[0])