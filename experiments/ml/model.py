import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from generate_data import generate_similar_data
import hashlib
from datetime import datetime
import os


def generate_model_filename():
    """
    현재 시간을 기반으로 SHA256 해시값을 생성하여 모델 파일명을 생성합니다.
    """
    # 현재 시간을 문자열로 변환
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # SHA256 해시 생성
    hash_object = hashlib.sha256(current_time.encode())
    hash_value = hash_object.hexdigest()[:10]  # 앞 10자리만 사용
    
    # 파일명 생성
    filename = f"lung_cancer_model_{hash_value}.cbm"
    return filename

# 1. 데이터 로드 및 기본 전처리
def load_and_preprocess_data(data):
    # 컬럼명 소문자로 변경
    data.columns = data.columns.str.lower()
    
    # LUNG_CANCER 타겟 변수를 이진값으로 변환
    le = LabelEncoder()
    data['lung_cancer'] = le.fit_transform(data['lung_cancer'])
    
    # gender 변수 이진값으로 변환
    data['gender'] = le.fit_transform(data['gender'])
    
    return data

# 2. 특성 중요도 시각화 함수
def plot_feature_importance(model, feature_names):
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance)
    plt.title('Feature Importance')
    plt.show()

# 3. 메인 모델링 파이프라인
def train_catboost_model(data, model_dir='models'):
    """
    CatBoost 모델을 학습하고 저장합니다.
    
    Parameters:
    - data: 학습할 데이터
    - model_dir: 모델을 저장할 디렉토리
    """
    # 데이터 전처리
    data = load_and_preprocess_data(data)
    
    # 특성과 타겟 분리
    X = data.drop(['lung_cancer'], axis=1)
    y = data['lung_cancer']
    
    # 학습/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # CatBoost 모델 정의
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.001,
        depth=6,
        loss_function='Logloss',
        random_seed=42,
        verbose=100
    )
    
    # 교차 검증 수행
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\n교차 검증 점수: {cv_scores}")
    print(f"평균 교차 검증 점수: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 전체 학습 데이터로 최종 모델 학습
    model.fit(X_train, y_train)
    
    # 테스트 세트로 예측
    y_pred = model.predict(X_test)
    
    # 모델 성능 평가
    print("\n분류 보고서:")
    print(classification_report(y_test, y_pred))
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # 특성 중요도 시각화
    plot_feature_importance(model, X.columns)
    
    # 모델 저장 디렉토리 생성
    os.makedirs(model_dir, exist_ok=True)
    
    # 시간 기반 해시를 포함한 파일명 생성
    model_filename = generate_model_filename()
    model_path = os.path.join(model_dir, model_filename)
    
    # 모델 저장
    model.save_model(model_path)
    print(f"\n모델이 저장되었습니다: {model_path}")
    
    return model, model_path


# 모델 로드 함수
def load_catboost_model(model_path):
    """
    저장된 CatBoost 모델을 불러옵니다.
    
    Parameters:
    - model_path: 저장된 모델의 경로
    
    Returns:
    - loaded_model: 불러온 CatBoost 모델
    """
    from catboost import CatBoostClassifier
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # 모델 불러오기
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    print(f"\n모델을 불러왔습니다: {model_path}")
    
    return loaded_model

# # 데이터 로드 및 모델 학습 실행
# data = pd.read_csv('data/lung_cancer/survey_lung_cancer.csv')
# model = train_catboost_model(data)

# 새로운 데이터에 대한 예측 함수
def predict_lung_cancer(model, new_data):
    """
    새로운 데이터에 대한 폐암 위험도를 예측합니다.
    
    Parameters:
    - model: 학습된 CatBoost 모델
    - new_data: 예측하고자 하는 새로운 데이터 (DataFrame)
    
    Returns:
    - 예측 결과 (0: 폐암 위험 낮음, 1: 폐암 위험 높음)
    - 예측 확률
    """
    new_data = load_and_preprocess_data(new_data.copy())
    if 'lung_cancer' in new_data.columns:
        new_data = new_data.drop(['lung_cancer'], axis=1)
    
    prediction = model.predict(new_data)
    probability = model.predict_proba(new_data)[:, 1]
    
    return prediction, probability


# # 원본 데이터 로드
# data = pd.read_csv('data/lung_cancer/survey_lung_cancer.csv')

# # 새로운 유사 데이터 10개 생성
# new_samples = generate_similar_data(data, n_samples=10)

# # 결과 출력
# print("\n생성된 새로운 데이터 샘플:")
# print(new_samples.to_string(index=False))
    
# pred, prob = predict_lung_cancer(model, new_samples)
# print(f"prediced value: {pred}")
# print(f"probability value: {prob}")

# 메인 실행 코드 수정
if __name__ == "__main__":
    # 데이터 로드 및 모델 학습
    data = pd.read_csv('data/lung_cancer/survey_lung_cancer.csv')
    
    # 모델 학습 및 저장
    model, model_path = train_catboost_model(data, 'models')
    
    # 새로운 유사 데이터 생성
    new_samples = generate_similar_data(data, n_samples=10)
    print(f"\n 새로운 데이터")
    print(new_samples)
    print(end="\n\n")
    
    # 저장된 모델 불러오기
    loaded_model = load_catboost_model(model_path)
    
    # 새로운 데이터로 예측
    pred, prob = predict_lung_cancer(loaded_model, new_samples)
    print(f"\n<예측 결과>")
    print(f"예측 클래스: {pred}")
    print(f"예측 확률: {prob}")