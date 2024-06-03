import pandas as pd

# 파일 경로 지정
file_path = r'C:\Users\82102\Desktop\전주시\건강검진정보_기준년도,일련번호제거.csv'

# CSV 파일 읽기
data = pd.read_csv(file_path, encoding='cp949')

# 신장 단위 변환 (cm에서 m로)
data['신장(m)'] = data['신장(5cm단위)'] / 100

# BMI 계산
data['BMI'] = data['체중(5kg단위)'] / (data['신장(m)'] ** 2)

# BMI 값을 소수점 둘째 자리까지 반올림
data['BMI'] = data['BMI'].round(2)

# BMI 열 추가
# 이미 계산해서 추가한 상태

# 결과를 새로운 CSV 파일로 저장
new_file_path = r'C:\Users\82102\Desktop\전주시\건강검진정보_BMI추가.csv'
data.to_csv(new_file_path, index=False, encoding='cp949')