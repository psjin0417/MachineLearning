import joblib
import numpy as np
import os
import time
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 설정 변수 ---
INPUT_FILE = "./dataset_features_color_2.pkl" 
MODEL_FILE = "./svm_model_pca_rbf.pkl" # 모델 저장명
RANDOM_STATE = 42

def train_pca_rbf():
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
    # RBF 커널은 스케일링이 필수적입니다.
    # PCA는 차원 축소를 통해 속도를 높이고 과적합을 방지합니다.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),       # 정규화 (평균0, 분산1) - 필수
        ('pca', PCA()),                     # 주성분 분석 (차원 축소)
        ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE))
    ])

    # 4. Grid Search 파라미터 설정
    # pca__n_components: 유지할 정보량(분산) 비율 (0.95 = 95%) 또는 개수
    # svm__C: 규제 강도 (클수록 곡선이 복잡해짐, 작을수록 단순해짐)
    # svm__gamma: RBF 커널의 폭 (클수록 좁고 뾰족함, 작을수록 넓고 완만함)
    param_grid = [
        {
            'pca__n_components': [0.95, 0.99], # 95% 또는 99% 정보 유지
            'svm__C': [1, 10, 100],
            'svm__gamma': [0.001, 0.01, 0.1, 'scale']
        }
    ]

    print("\n>>> Grid Search (PCA + RBF SVM) 시작... (시간이 걸릴 수 있습니다)")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=3,                 # 교차 검증 3회
        scoring='accuracy', 
        verbose=2, 
        n_jobs=1              # [수정] 메모리 오류 방지를 위해 병렬 처리 끔 (n_jobs=-1 -> 1)
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()

    print(f"\n>>> 학습 완료! 소요 시간: {end_time - start_time:.2f}초")
    print(f"최적의 파라미터: {grid_search.best_params_}")
    print(f"최고 교차 검증 점수: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    # PCA로 얼마나 차원이 줄었는지 확인
    pca_step = best_model.named_steps['pca']
    print(f"\n[PCA 결과] {X.shape[1]} 차원 -> {pca_step.n_components_} 차원으로 축소됨 (정보 보존율 설정에 따름)")

    # 5. 성능 평가
    print("\n" + "="*40)
    print(">>> 테스트 세트 성능 평가")
    print("="*40)

    # 6. 속도 성능 측정
    predict_start = time.time()
    y_pred = best_model.predict(X_test)
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
    joblib.dump(best_model, MODEL_FILE)
    print(f"\n모델 저장 완료: {MODEL_FILE}")

if __name__ == "__main__":
    train_pca_rbf()
