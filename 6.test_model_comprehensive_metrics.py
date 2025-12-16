import cv2
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_score,
    recall_score,
    f1_score
)
import seaborn as sns
import time

# --- 설정 변수 ---
# 테스트 데이터셋 경로
TEST_POS_DIR = "./test/positive"
TEST_NEG_DIR = "./test/negative"
MODEL_PATH = "./svm_model_v3_gridsearch.pkl"
TARGET_SIZE = (128, 128)

# HOG 파라미터 (학습과 동일해야 함)
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (16, 16),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'visualize': False,
    'transform_sqrt': True,
}

# 컬러 히스토그램 파라미터
HIST_BINS = (32, 32)

def extract_color_histogram(image, bins=(32, 32)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, bins, [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    # 1. 리사이즈
    img_resized = cv2.resize(img, TARGET_SIZE)
    
    # 2. HOG
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_feature = hog(gray, **HOG_PARAMS)
    
    # 3. Color
    color_feature = extract_color_histogram(img_resized, bins=HIST_BINS)
    
    # 4. 결합
    return np.hstack([hog_feature, color_feature])

def load_test_data():
    X_test = []
    y_test = []
    
    # Positive (Label 1)
    print(f"Loading Positive samples from {TEST_POS_DIR}...")
    pos_files = [f for f in os.listdir(TEST_POS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for f in pos_files:
        feat = extract_features(os.path.join(TEST_POS_DIR, f))
        if feat is not None:
            X_test.append(feat)
            y_test.append(1)
            
    # Negative (Label 0)
    print(f"Loading Negative samples from {TEST_NEG_DIR}...")
    neg_files = [f for f in os.listdir(TEST_NEG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for f in neg_files:
        feat = extract_features(os.path.join(TEST_NEG_DIR, f))
        if feat is not None:
            X_test.append(feat)
            y_test.append(0)
            
    return np.array(X_test), np.array(y_test)

def evaluate_model():
    # 1. 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return
        
    print(f">>> Loading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    
    # 2. 데이터 로드 및 특징 추출
    X_test, y_test = load_test_data()
    print(f"Total Test Samples: {len(X_test)}")
    if len(X_test) == 0:
        print("Error: No test data found.")
        return

    # 3. 예측
    print(">>> Predicting...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / len(X_test)) * 1000
    
    print(f"\n[Inference Time]")
    print(f"Total Time for {len(X_test)} images : {total_time:.4f} sec")
    print(f"Average Time per Image         : {avg_time_ms:.4f} ms")

    y_prob = model.predict_proba(X_test)[:, 1] # Positive class probability
    
    # 4. 성능 지표 계산
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MATCHINE LEARNING MODEL EVALUATION REPORT")
    print("="*50)
    
    print(f"\n[Basic Metrics]")
    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    
    print(f"\n[Classification Report]")
    print(classification_report(y_test, y_pred, target_names=['Background (Neg)', 'Kickboard (Pos)']))
    
    # 5. 혼동 행렬 (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n[Confusion Matrix Analysis]")
    print(f"Total Samples: {len(y_test)}")
    print(f"- True Negatives (Correct Background) : {tn}")
    print(f"- False Positives (Mis-Detection)       : {fp}  <-- 줄여야 할 오탐지")
    print(f"- False Negatives (Missed Kickboard)  : {fn}  <-- 위험한 미탐지")
    print(f"- True Positives (Correct Kickboard)  : {tp}")

    # 6. ROC Curve & AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"\n[ROC/AUC]")
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # 7. 실제 파이프라인(Feature Extraction + Prediction) 속도 측정
    print("\n" + "="*50)
    print("FULL PIPELINE SPEED BENCHMARK")
    print("="*50)
    
    if len(X_test) > 0:
        # 테스트용 이미지 하나 로드 (Positive 폴더의 첫번째 파일)
        sample_file = os.listdir(TEST_POS_DIR)[0]
        sample_path = os.path.join(TEST_POS_DIR, sample_file)
        sample_img = cv2.imread(sample_path)
        
        if sample_img is not None:
            # 워밍업
            _ = extract_features(sample_path)
            
            # A. Feature Extraction 시간 측정
            fe_start = time.time()
            cnt = 100
            for _ in range(cnt):
                # 실제 슬라이딩 윈도우에서는 이미지가 메모리에 잇으므로 extract_features_single_image 처럼 동작
                # 여기서는 파일 로드 제외하고 resize+hog+color 부분만 측정해야 정확함
                # extract_features 함수는 파일 로드를 포함하므로, 함수 내부 로직을 분리하거나 이미지를 인자로 넘기는게 맞음
                # 간단히 extract_features 함수를 수정하지 않고, 로직을 직접 수행
                
                # 1. 리사이즈
                img_resized = cv2.resize(sample_img, TARGET_SIZE)
                # 2. HOG
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                hog_feat = hog(gray, **HOG_PARAMS)
                # 3. Color
                hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, HIST_BINS, [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                color_feat = hist.flatten()
                # 4. 결합
                feat = np.hstack([hog_feat, color_feat]).reshape(1, -1)
                
            fe_end = time.time()
            avg_fe_time_ms = ((fe_end - fe_start) / cnt) * 1000
            
            # B. Prediction 시간 측정 (위의 feat 사용)
            pred_start = time.time()
            for _ in range(cnt):
                _ = model.predict(feat)
            pred_end = time.time()
            avg_pred_time_ms = ((pred_end - pred_start) / cnt) * 1000
            
            total_latency = avg_fe_time_ms + avg_pred_time_ms
            
            print(f"Feature Extraction Time : {avg_fe_time_ms:.4f} ms")
            print(f"SVM Prediction Time     : {avg_pred_time_ms:.4f} ms")
            print(f"-"*30)
            print(f"TOTAL Latency per Window: {total_latency:.4f} ms")
            
            fps_est = 1000 / total_latency
            print(f"Theoretical Max FPS     : {fps_est:.1f} FPS (Single Thread)")
    
evaluate_model()
