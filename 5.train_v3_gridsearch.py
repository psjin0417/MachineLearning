import joblib
import numpy as np
import os
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 설정 변수 ---
INPUT_FILE = "./dataset_features_color_2.pkl" 
MODEL_FILE = "./svm_model_v3_gridsearch.pkl" # v3로 파일명 변경

def train_v3_gridsearch():
    # 1. 데이터 로드
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} 파일이 없습니다. feature extraction 코드를 먼저 실행하세요.")
        return

    print(f">>> [{INPUT_FILE}] 데이터 로딩 중...")
    data = joblib.load(INPUT_FILE)
    X = data['X']
    y = data['y']

    print(f"데이터 로드 완료. 차원(Features): {X.shape[1]}, 샘플 수: {X.shape[0]}")

    # 2. 학습/테스트 분리 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")

    # 3. SVM 모델 및 Grid Search 설정
    print("\n>>> 최적의 파라미터 찾기 (Grid Search) 시작...")
    
    # 테스트할 파라미터 조합 설정
    # C: 규제 강도 (작을수록 규제 강함/둔감, 클수록 규제 약함/민감)
    # kernel: 'linear'가 속도가 가장 빠름. 'rbf'는 정확도는 높으나 느림.
    param_grid = [
        {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]},
        # 만약 속도가 조금 느려도 정확도가 중요하다면 아래 주석 해제하여 rbf도 테스트 가능
        # {'kernel': ['rbf'], 'C': [1, 10, 100], 'gamma': [0.01, 0.001]} 
    ]

    # 기본 SVM 설정 (확률 계산 가능, 밸런스 조정)
    svm_base = SVC(probability=True, class_weight='balanced', random_state=42)

    # GridSearchCV 설정 (cv=5: 교차 검증 5번 수행)
    grid_search = GridSearchCV(
        svm_base, 
        param_grid, 
        cv=5, 
        scoring='accuracy', 
        verbose=2, 
        n_jobs=-1 # 가능한 모든 CPU 코어 사용
    )

    train_start = time.time()
    grid_search.fit(X_train, y_train)
    train_end = time.time()

    print(f"\n>>> Grid Search 완료! 소요 시간: {train_end - train_start:.2f}초")
    print(f"최적의 파라미터: {grid_search.best_params_}")
    print(f"최고 교차 검증 점수: {grid_search.best_score_:.4f}")

    # 최적의 모델 가져오기
    best_svm = grid_search.best_estimator_

    # ====================================================
    # 4. 성능 평가 (최적 모델 사용)
    # ====================================================
    print("\n" + "="*40)
    print(">>> 최적 모델(Best Model) 성능 정밀 분석 결과")
    print("="*40)

    # A. 속도 성능 측정
    predict_start = time.time()
    y_pred = best_svm.predict(X_test)
    predict_end = time.time()

    total_time = predict_end - predict_start
    avg_time_ms = (total_time / len(X_test)) * 1000
    fps = 1000 / avg_time_ms if avg_time_ms > 0 else 0

    print(f"[속도 성능]")
    print(f"- 테스트 샘플 {len(X_test)}개 전체 예측 시간: {total_time:.4f}초")
    print(f"- 샘플 1개당 평균 예측 시간: {avg_time_ms:.3f} ms")
    print(f"- 예상 처리 속도: {fps:.1f} FPS")

    # B. 정확도 성능 측정
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[정확도 성능]")
    print(f"- 전체 정확도 (Accuracy): {acc * 100:.2f}%")
    
    print("\n[상세 리포트]")
    print(classification_report(y_test, y_pred, target_names=['Background', 'Vehicle']))

    # C. 오차 행렬
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n[오차 행렬 분석]")
    print(f"[[ TN({tn})\t FP({fp}) ]")  
    print(f" [ FN({fn})\t TP({tp}) ]]")
    
    # 5. 모델 저장
    joblib.dump(best_svm, MODEL_FILE)
    print(f"\n최적 모델 저장 완료: {MODEL_FILE}")

if __name__ == "__main__":
    train_v3_gridsearch()
