import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# --- 설정 변수 ---
INPUT_FILE = "./dataset_features.pkl"
MODEL_FILE = "./svm_model.pkl"

def train():
    # 1. 데이터 로드
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} 파일이 없습니다. extract_features.py를 먼저 실행하세요.")
        return

    print(">>> 데이터 로딩 중...")
    data = joblib.load(INPUT_FILE)
    X = data['X']
    y = data['y']

    print(f"데이터 로드 완료. X shape: {X.shape}, y shape: {y.shape}")

    # 2. 학습 데이터와 테스트 데이터 분리 (8:2 비율)
    # stratify=y 옵션: Positive/Negative 비율을 유지하며 나눔
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")

    # 3. SVM 모델 생성 및 학습
    # kernel='linear': 속도가 빠르고 HOG 같은 고차원 특징에 잘 작동함
    # kernel='rbf': 비선형 문제에 좋지만 파라미터 튜닝이 필요하고 느림
    #svm = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    #svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm = SVC(kernel='linear', C=1.0, probability=True, 
          class_weight='balanced', random_state=42)


    print(">>> SVM 학습 시작 (데이터 양에 따라 시간이 걸릴 수 있습니다)...")
    start_time = time.time()
    svm.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"학습 완료! 소요 시간: {end_time - start_time:.2f}초")

    # 4. 모델 평가
    print("\n>>> 테스트 데이터 평가 결과:")
    y_pred = svm.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"정확도(Accuracy): {acc * 100:.2f}%")
    
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # 5. 모델 저장
    joblib.dump(svm, MODEL_FILE)
    print(f"모델 저장 완료: {MODEL_FILE}")

if __name__ == "__main__":
    import os
    train()