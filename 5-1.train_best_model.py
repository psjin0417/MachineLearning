import joblib
import numpy as np
import os
import time
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- 설정 변수 ---
INPUT_FILE = "./dataset_features_color_2.pkl"
MODEL_FILE = "./svm_model_pca_rbf.pkl" # 모델 저장명 (기존 파일 덮어쓰기 주의)
RANDOM_STATE = 42

def train_best_model():
    # 1. 데이터 로드
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} 파일이 없습니다.")
        return

    print(f">>> [{INPUT_FILE}] 데이터 로딩 중...")
    data = joblib.load(INPUT_FILE)
    X = data['X']
    y = data['y']

    print(f"원본 데이터 차원: {X.shape[1]}, 샘플 수: {X.shape[0]}")

    # 2. 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")

    # 3. 파이프라인 구축 (Scaler -> PCA -> SVM)
    # 최적 파라미터 적용: pca__n_components=0.95, svm__C=10, svm__gamma='scale'
    print("\n>>> 학습 파이프라인 구축 중...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),       # 정규화
        ('pca', PCA(n_components=0.95)),    # 최적화된 PCA 차원 축소 비율
        ('svm', SVC(C=10,                   # 최적화된 C 값
                   gamma='scale',           # 최적화된 gamma 값
                   kernel='rbf', 
                   probability=True, 
                   class_weight='balanced', 
                   random_state=RANDOM_STATE))
    ])

    # 4. 모델 학습
    print(f"\n>>> 모델 학습 시작 (C=10, gamma='scale', PCA=0.95)...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    end_time = time.time()

    print(f"\n>>> 학습 완료! 소요 시간: {end_time - start_time:.2f}초")

    # PCA로 얼마나 차원이 줄었는지 확인
    pca_step = pipeline.named_steps['pca']
    print(f"\n[PCA 결과] {X.shape[1]} 차원 -> {pca_step.n_components_} 차원으로 축소됨 (정보 보존율 95%)")

    # 5. 성능 평가
    print("\n" + "="*40)
    print(">>> 테스트 세트 성능 평가")
    print("="*40)

    # 6. 속도 성능 측정
    predict_start = time.time()
    y_pred = pipeline.predict(X_test)
    predict_end = time.time()
    
    total_time = predict_end - predict_start
    avg_time_ms = (total_time / len(X_test)) * 1000
    
    print(f"[예측 속도]")
    print(f"- 평균 예측 시간 (Scaling+PCA+SVM): {avg_time_ms:.3f} ms/sample")
    
    # 정확도 성능
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[정확도 성능]")
    print(f"- Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Background', 'Vehicle']))

    # 모델 저장
    joblib.dump(pipeline, MODEL_FILE)
    print(f"\n모델 저장 완료: {MODEL_FILE}")

if __name__ == "__main__":
    train_best_model()
