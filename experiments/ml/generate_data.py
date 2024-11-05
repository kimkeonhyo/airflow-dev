import pandas as pd
import numpy as np
import random

def generate_similar_data(original_data, n_samples=10):
    """
    원본 데이터의 특성을 보존하면서 유사한 새로운 데이터를 생성합니다.
    """
    # 데이터 복사
    data = original_data.copy()
    
    # 컬럼명을 소문자로 변경 (원본은 유지)
    data.columns = data.columns.str.lower().str.strip()
    original_columns = original_data.columns
    
    # 이진 특성과 연속형 특성 구분
    binary_cols = [col for col in data.columns if col != 'age' and col != 'lung_cancer']
    continuous_cols = ['age']
    
    # 각 특성의 통계량 계산
    stats = {
        'binary': {col: data[col].value_counts(normalize=True) for col in binary_cols},
        'continuous': {
            'age': {
                'mean': data['age'].mean(),
                'std': data['age'].std(),
                'min': data['age'].min(),
                'max': data['age'].max()
            }
        }
    }
    
    # 새로운 데이터 생성
    new_data = []
    for _ in range(n_samples):
        sample = {}
        
        # 이진 특성 생성
        for col in binary_cols:
            col_stripped = col.strip()
            sample[col_stripped] = np.random.choice(
                list(stats['binary'][col].index),
                p=list(stats['binary'][col].values)
            )
        
        # 연령 생성 (정규분포 사용, 실제 범위 내로 제한)
        age = int(np.clip(
            np.random.normal(
                stats['continuous']['age']['mean'],
                stats['continuous']['age']['std']
            ),
            stats['continuous']['age']['min'],
            stats['continuous']['age']['max']
        ))
        sample['age'] = age
        
        # LUNG_CANCER 결과 생성 (조건부 확률 기반)
        age_group = 'old' if age >= 60 else 'young'
        smoking_status = 'smoker' if sample['smoking'] == 2 else 'non_smoker'
        
        # 주요 위험 요인 수 계산 (소문자로 된 컬럼명 사용)
        risk_factors = sum([
            sample['yellow_fingers'] == 2,
            sample['anxiety'] == 2,
            sample['shortness of breath'] == 2,
            sample['chest pain'] == 2,
            sample['coughing'] == 2,
            sample['wheezing'] == 2
        ])
        
        # 위험 요인 기반 폐암 확률 계산
        cancer_prob = 0.3  # 기본 확률
        cancer_prob += 0.1 if age_group == 'old' else 0
        cancer_prob += 0.15 if smoking_status == 'smoker' else 0
        cancer_prob += 0.05 * risk_factors
        cancer_prob = min(cancer_prob, 0.9)  # 최대 90% 확률로 제한
        
        sample['lung_cancer'] = 'YES' if random.random() < cancer_prob else 'NO'
        
        # 원본 컬럼명과 동일한 형태로 sample 딕셔너리 키 설정
        new_sample = {col: sample[col.lower().strip()] for col in original_columns}
        new_data.append(new_sample)
    
    # DataFrame으로 변환 (원본 컬럼명 사용)
    result_df = pd.DataFrame(new_data, columns=original_columns)
    
    return result_df

# # 사용 예시:
# if __name__ == "__main__":
#     # 원본 데이터 로드
#     data = pd.read_csv('data/lung_cancer/survey_lung_cancer.csv')
    
#     # 새로운 유사 데이터 10개 생성
#     new_samples = generate_similar_data(data, n_samples=10)
    
#     # 결과 출력
#     print("\n생성된 새로운 데이터 샘플:")
#     print(new_samples.to_string(index=False))