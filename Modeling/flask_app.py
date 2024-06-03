from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # CORS 활성화

# 저장된 모델 로드
calibrated_model = joblib.load('calibrated_model.pkl')
scaler, pca, knn = joblib.load('health_model.pkl')

@app.route('/predict_calibrated', methods=['POST'])
def predict_calibrated():
    data = request.json
    input_data = pd.DataFrame([data])
    prob = calibrated_model.predict_proba(input_data)[:, 1][0]
    return jsonify({'label_1_probability': round(prob, 4)})

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    data = request.json
    input_data = pd.DataFrame([{
        '성별': data.get('gender'),
        '연령대코드(5세단위)': data.get('age'),
        '수축기혈압': data.get('systolicBloodPressure'),
        '이완기혈압': data.get('diastolicBloodPressure'),
        'BMI': data.get('bmi')
    }])
    
    # 스케일링 및 PCA 변환 적용
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    
    # 예측 수행
    prob = knn.predict_proba(input_pca)[:, 1][0]
    
    # 점수 계산
    score = 100 - prob * 100
    score += 2 if data.get('smokingStatus') == '오늘 피우지 않았다' else -2
    score += 2 if data.get('alcoholConsumption') == '오늘 마시지 않았다' else -2

    # 점수에 따른 메시지 결정
    if score < 20:
        message = "오늘부터 매일 30분씩 산책을 시작해보세요"
    elif score < 40:
        message = "일주일에 3번, 산책하는 습관을 가져봐요"
    elif score < 60:
        message = "건강을 위해 걷는 시간을 조금 늘려봐요"
    elif score < 80:
        message = "더 나은 건강을 위해 꾸준히 걸어봐요"
    else:
        message = "훌륭해요! 계속해서 지금의 건강을 유지하세요"

    return jsonify({'label_1_probability': round(prob, 2), 'score': round(score, 2), 'message': message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)