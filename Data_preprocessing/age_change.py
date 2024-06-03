import pandas as pd

# 파일 경로
file_path = 'C:/Users/82102/Desktop/전주시/건강검진정보_null제거.csv'

# 연령대 코드와 중간 연령 매핑
age_mapping = {
    1: 2, 2: 7, 3: 12, 4: 17, 5: 22, 6: 27, 7: 32, 8: 37, 9: 42,
    10: 47, 11: 52, 12: 57, 13: 62, 14: 67, 15: 72, 16: 77, 17: 82, 18: 87
}

# CSV 파일 읽기
data = pd.read_csv(file_path, encoding='cp949')

# 연령대 코드를 중간값으로 변환
data['연령대코드(5세단위)'] = data['연령대코드(5세단위)'].map(age_mapping)

# 결과를 새로운 CSV 파일로 저장
new_file_path = 'C:/Users/82102/Desktop/전주시/건강검진정보_나이변경.csv'
data.to_csv(new_file_path, index=False, encoding='cp949')