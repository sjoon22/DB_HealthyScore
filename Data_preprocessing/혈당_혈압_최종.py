import pandas as pd

# 파일 경로
file_path = 'C:/Users/82102/Desktop/전주시/modified_data.csv'

# CSV 파일 읽기
data = pd.read_csv(file_path)

# 'DIS' 값을 조건에 따라 변경
# 1, 2, 3이면 1로, 4이면 0으로
data['DIS'] = data['DIS'].apply(lambda x: 1 if x in [1, 2, 3] else 0)

# 변경된 데이터를 새로운 CSV 파일로 저장
new_file_path = 'C:/Users/82102/Desktop/전주시/혈당_혈압_최종.csv'
data.to_csv(new_file_path, index=False, encoding='cp949')

print("파일 저장 완료: ", new_file_path)