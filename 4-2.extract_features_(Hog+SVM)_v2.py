import cv2
import os
import numpy as np
import joblib
from skimage.feature import hog

# --- 설정 변수 ---
POS_DATA_DIR = "./positive_aug"
NEG_DATA_DIR = "./negative_aug"
OUTPUT_FILE = "./dataset_features_color.pkl" # 파일명 변경

# HOG 파라미터
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (16, 16), # 차원을 줄이기 위해 16x16 추천
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'visualize': False,
    'transform_sqrt': True
}

# 컬러 히스토그램 파라미터
HIST_BINS = (32, 32) # Hue(색상) 32단계, Saturation(채도) 32단계로 구분

def extract_color_histogram(image, bins=(32, 32)):
    """
    이미지에서 색상 정보를 추출하여 벡터로 만듭니다.
    RGB 대신 HSV를 사용하는 것이 조명 변화에 더 강합니다.
    """
    # 1. HSV 색상 공간으로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 2. 히스토그램 계산 (H와 S 채널만 사용, V(명도)는 그림자 때문에 제외하는 게 보통)
    # ranges: H(0~180), S(0~256)
    hist = cv2.calcHist([hsv], [0, 1], None, bins, [0, 180, 0, 256])
    
    # 3. 정규화 (이미지 크기에 상관없이 비율로 계산)
    cv2.normalize(hist, hist)
    
    # 4. 1차원 벡터로 펼치기 (flatten)
    return hist.flatten()

def load_images_and_extract_features(folder_path, label):
    features = []
    labels = []
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    print(f"[{folder_path}]에서 {len(files)}장의 이미지를 처리합니다...")

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath)
        
        if img is None:
            continue

        # 1. 이미지 리사이즈 (128x128)
        img = cv2.resize(img, (128, 128))
        
        # --- A. HOG 특징 추출 (형태) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_feature = hog(gray, **HOG_PARAMS)
        
        # --- B. 컬러 히스토그램 추출 (색상) ---
        color_feature = extract_color_histogram(img, bins=HIST_BINS)
        
        # --- C. 특징 결합 (Concatenate) ---
        # [HOG 특징들 ... , Color 특징들 ...] 순서로 이어 붙임
        combined_feature = np.hstack([hog_feature, color_feature])
        
        features.append(combined_feature)
        labels.append(label)

    return features, labels

if __name__ == "__main__":
    print(">>> HOG + Color 특징 추출 시작")

    # Positive 데이터 (Label: 1)
    pos_features, pos_labels = load_images_and_extract_features(POS_DATA_DIR, 1)
    
    # Negative 데이터 (Label: 0)
    neg_features, neg_labels = load_images_and_extract_features(NEG_DATA_DIR, 0)

    if not pos_features or not neg_features:
        print("Error: 데이터가 부족합니다.")
        exit()

    # 데이터 합치기
    X = np.array(pos_features + neg_features)
    y = np.array(pos_labels + neg_labels)

    print("------------------------------------------------")
    print(f"전체 데이터 개수: {len(X)}")
    
    # HOG 길이 + Color 길이 확인
    hog_len = len(hog(np.zeros((128,128)), **HOG_PARAMS))
    color_len = HIST_BINS[0] * HIST_BINS[1]
    print(f"특징 벡터 길이: {X.shape[1]} (HOG: {hog_len} + Color: {color_len})")

    data_to_save = {'X': X, 'y': y}
    joblib.dump(data_to_save, OUTPUT_FILE)
    
    print(f"데이터 저장 완료: {OUTPUT_FILE}")