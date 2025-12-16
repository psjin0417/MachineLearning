import cv2
import os
import numpy as np
import time
import joblib
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. 설정 및 파라미터 ---
# [중요] 이 파라미터들은 데이터셋(PKL)을 생성할 때 사용한 값과 '정확히' 일치해야 합니다.
# 불일치 시 차원 오류가 발생합니다.
IMG_SIZE = (128, 128)

# HOG 파라미터 (dataset_features_color_2.pkl 생성 설정으로 추정값)
# 만약 차원 에러가 나면 이 값들을 수정해야 합니다 (예: pixels_per_cell=(8,8) 등)
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (16, 16), # v2 코드 기준 (16, 16). v1은 (8, 8)
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'visualize': False,
    'transform_sqrt': True
}

# 컬러 히스토그램 파라미터
USE_COLOR_HIST = True   # Color 포함 여부 (파일명에 'color'가 있으므로 True 추정)
HIST_BINS = (32, 32)    # (H_bins, S_bins)

# 데이터 경로
TRAIN_DATA_FILE = "./dataset_features_color_3.pkl"
TEST_POS_DIR = "./test/positive"
TEST_NEG_DIR = "./test/negative"
MODEL_SAVE_PATH = "./svm_model_comprehensive.pkl"

# 학습 파라미터
PCA_COMPONENTS = 0.7
SVM_PARAMS = {
    'C': 10,
    'gamma': 'scale',
    'kernel': 'rbf',
    'class_weight': 'balanced',
    'probability': True,
    'random_state': 42
}

def extract_features(image):
    """단일 이미지에 대해 HOG (+컬러) 특징을 추출합니다."""
    # 0. 리사이즈
    img_resized = cv2.resize(image, IMG_SIZE)
    
    # 1. HOG 특징
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, **HOG_PARAMS)
    
    # 2. 컬러 히스토그램 (옵션)
    if USE_COLOR_HIST:
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        # H, S 채널에 대한 히스토그램
        hist = cv2.calcHist([hsv], [0, 1], None, HIST_BINS, [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        color_feat = hist.flatten()
        
        # 결합
        final_feat = np.hstack([hog_feat, color_feat])
    else:
        final_feat = hog_feat
        
    return final_feat

def load_train_data_from_pkl(pkl_path):
    print(f"\n>>> [학습 데이터] PKL 파일 로드 중: {pkl_path}")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {pkl_path}")
        
    data = joblib.load(pkl_path)
    X = data['X']
    y = data['y']
    print(f"  => 로드 완료. 데이터 개수: {len(X)}")
    print(f"  => 데이터 차원: {X.shape[1]}")
    return X, y

def load_test_data_from_images(pos_dir, neg_dir):
    print(f"\n>>> [테스트 데이터] 이미지파일 로드 및 특징 추출 중...")
    X = []
    y = []
    total_time_ms = 0
    count = 0
    
    # Positive
    for folder, label in [(pos_dir, 1), (neg_dir, 0)]:
        if not os.path.exists(folder):
            print(f"  [Warning] 폴더 없음: {folder}")
            continue
            
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
        print(f"  - {folder}: {len(files)}장 처리 중...")
        
        for f in files:
            path = os.path.join(folder, f)
            img = cv2.imread(path)
            if img is None: continue
            
            # 시간 측정
            start = time.perf_counter()
            feat = extract_features(img)
            end = time.perf_counter()
            
            total_time_ms += (end - start) * 1000
            X.append(feat)
            y.append(label)
            count += 1
            
    if count == 0:
        return np.array([]), np.array([]), 0
        
    avg_time = total_time_ms / count
    print(f"  => 테스트 데이터 추출 완료 ({count}장)")
    print(f"  => 평균 특징 추출 시간: {avg_time:.3f} ms")
    return np.array(X), np.array(y), avg_time

def main():
    print("="*60)
    print("      종합 모델 학습 및 성능 평가 (PKL Train + Image Test)      ")
    print("="*60)
    
    # 1. 학습 데이터 로드 (PKL)
    try:
        X_train, y_train = load_train_data_from_pkl(TRAIN_DATA_FILE)
    except Exception as e:
        print(f"[Error] 학습 데이터 로드 실패: {e}")
        return

    # [Validation] 차원 검증
    # 현재 설정(HOG_PARAMS)으로 더미 이미지 하나를 추출해보고 차원이 맞는지 확인
    dummy_img = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
    dummy_feat = extract_features(dummy_img)
    expected_dim = len(dummy_feat)
    
    print("-" * 50)
    print(f"PKL 데이터 차원: {X_train.shape[1]}")
    print(f"현재 설정 예상 차원: {expected_dim}")
    
    if X_train.shape[1] != expected_dim:
        print("\n" + "!"*60)
        print("[CRITICAL ERROR] 차원 불일치 발생!")
        print("불러온 PKL 파일의 특징 벡터 길이와 현재 스크립트 설정으로 추출한 길이가 다릅니다.")
        print(f"  - PKL: {X_train.shape[1]}")
        print(f"  - Script: {expected_dim}")
        print("해결 방법:")
        print("1. HOG_PARAMS의 pixels_per_cell 등을 수정하여 PKL 생성 시 설정과 맞추세요.")
        print("2. 또는 Feature Extraction 코드를 실행하여 PKL을 새로 생성하세요.")
        print("!"*60)
        return # 중단
        
    print(">>> 차원 검증 성공! 학습을 진행합니다.")

    # 2. 모델 학습
    print("\n>>> 학습 파이프라인 구축 및 학습 시작...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=PCA_COMPONENTS)),
        ('svm', SVC(**SVM_PARAMS))
    ])
    
    train_start = time.time()
    pipeline.fit(X_train, y_train)
    train_end = time.time()
    
    print(f"  => 학습 완료. 소요 시간: {train_end - train_start:.2f}초")
    print(f"  => PCA 적용 후 차원: {pipeline.named_steps['pca'].n_components_}")

    # 3. 테스트 데이터 로드 및 예측
    X_test, y_test, feat_time = load_test_data_from_images(TEST_POS_DIR, TEST_NEG_DIR)
    
    if len(X_test) == 0:
        print("[Error] 테스트 데이터가 없어 평가를 중단합니다.")
        return

    print("\n>>> 테스트 수행 (예측)...")
    pred_start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    pred_end = time.perf_counter()
    
    avg_pred_time = (pred_end - pred_start) * 1000 / len(X_test)

    # 4. 결과 출력
    print("\n" + "="*50)
    print("              최종 성능 리포트              ")
    print("="*50)
    
    # 시간
    print("[1] 속도 성능 (Speed)")
    print(f"  - 특징 추출 (HOG+Color): {feat_time:.3f} ms")
    print(f"  - SVM 예측 (PCA포함): {avg_pred_time:.3f} ms")
    print(f"  - **Total Pipeline**: {feat_time + avg_pred_time:.3f} ms/sample")
    
    # 정확도
    acc = accuracy_score(y_test, y_pred)
    print("\n[2] 모델 정확도 (Accuracy)")
    print(f"  - Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Background', 'Kickboard']))
    
    # Confusion Matrix
    print("\n[3] Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    # 2x2일 경우 예쁘게 출력
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"      | Pred 0 | Pred 1 |")
        print(f"True 0| {tn:6d} | {fp:6d} |")
        print(f"True 1| {fn:6d} | {tp:6d} |")
    else:
        print(cm)
        
    # 모델 저장
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    print(f"\n>>> 모델 저장 완료: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
