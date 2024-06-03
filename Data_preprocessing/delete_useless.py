import pandas as pd

# 파일 경로 지정
file_path = r'C:\Users\82102\Desktop\전주시\건강검진정보_나이변경.csv'

# CSV 파일 읽기
data = pd.read_csv(file_path, encoding='cp949')  # 'cp949' 인코딩을 가정, 필요에 따라 변경

# '기준년도', '가입자일련번호' 열 제거
data = data.drop(columns=['기준년도', '가입자일련번호'])

# 결과를 새로운 CSV 파일로 저장
new_file_path = r'C:\Users\82102\Desktop\전주시\건강검진정보_기준년도,일련번호제거.csv'
data.to_csv(new_file_path, index=False, encoding='cp949')  # 'cp949' 인코딩을 유지