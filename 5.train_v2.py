import joblib
import numpy as np
import os
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 설정 변수 ---
# 이전 단계에서 만든 파일명으로 변경 (HOG+Color라면 _color.pkl 일 수 있음)
INPUT_FILE = "./dataset_features_color.pkl" 
MODEL_FILE = "./svm_model_v2.pkl"

def train_v2():
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

    # 3. SVM 모델 설정
    # 실시간성이 중요하므로 kernel='linear' 추천
    # class_weight='balanced': 데이터 불균형(차가 적고 배경이 많을 때) 자동 보정
    svm = SVC(kernel='linear', C=1.0, probability=True, 
              class_weight='balanced', random_state=42)

    print("\n>>> SVM 학습 시작...")
    train_start = time.time()
    svm.fit(X_train, y_train)
    train_end = time.time()
    print(f"학습 완료! 소요 시간: {train_end - train_start:.2f}초")

    # ====================================================
    # 4. 성능 평가 (속도 및 정밀도 상세 분석) - v2 핵심 추가
    # ====================================================
    print("\n" + "="*40)
    print(">>> 모델 성능 정밀 분석 결과")
    print("="*40)

    # A. 속도 성능 측정 (Inference Time)
    predict_start = time.time()
    y_pred = svm.predict(X_test)
    predict_end = time.time()

    total_time = predict_end - predict_start
    avg_time_ms = (total_time / len(X_test)) * 1000
    fps = 1000 / avg_time_ms if avg_time_ms > 0 else 0

    print(f"[속도 성능]")
    print(f"- 테스트 샘플 {len(X_test)}개 전체 예측 시간: {total_time:.4f}초")
    print(f"- 샘플 1개당 평균 예측 시간: {avg_time_ms:.3f} ms")
    print(f"- 예상 처리 속도 (모델 연산만): {fps:.1f} FPS")
    print("  * 주의: 특징 추출(HOG+Color) 시간은 제외된 순수 분류 시간입니다.")

    # B. 정확도 성능 측정
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[정확도 성능]")
    print(f"- 전체 정확도 (Accuracy): {acc * 100:.2f}%")
    
    print("\n[상세 리포트 (Precision & Recall)]")
    # target_names: 0=Negative(배경), 1=Positive(차량)
    print(classification_report(y_test, y_pred, target_names=['Background', 'Vehicle']))

    # C. 오차 행렬 (Confusion Matrix) 상세
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n[오차 행렬 분석 (Confusion Matrix)]")
    print(f"[[ TN({tn})\t FP({fp}) ]")  
    print(f" [ FN({fn})\t TP({tp}) ]]")
    print("-" * 30)
    print(f"1. TN (True Negative): 배경을 배경이라고 잘 맞춤 -> {tn}개")
    print(f"2. FP (False Positive): 배경인데 차라고 오해함 (오탐) -> {fp}개")
    print(f"3. FN (False Negative): 차가 있는데 놓침 (미탐, 위험!) -> {fn}개")
    print(f"4. TP (True Positive): 차를 차라고 잘 맞춤 -> {tp}개")

    # 5. 모델 저장
    joblib.dump(svm, MODEL_FILE)
    print(f"\n모델 저장 완료: {MODEL_FILE}")

if __name__ == "__main__":
    train_v2()