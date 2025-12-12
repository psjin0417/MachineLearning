import cv2
import os
import numpy as np
import joblib
from skimage.feature import hog

# --- 설정 변수 ---
POS_DATA_DIR = "./positive_aug"
NEG_DATA_DIR = "./negative_aug"
OUTPUT_FILE = "./dataset_features.pkl" # 특징 벡터가 저장될 파일

# HOG 파라미터 설정 (중요!)
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'visualize': False,
    'transform_sqrt': True
}

def load_images_and_extract_hog(folder_path, label):
    features = []
    labels = []
    
    # 폴더 내의 모든 jpg 파일 읽기
    files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    print(f"[{folder_path}]에서 {len(files)}장의 이미지를 처리합니다...")

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath)
        
        if img is None:
            continue

        # 1. 흑백 변환 (HOG는 그레이스케일에서 작동)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. 이미지 리사이즈 (이미 라벨링 툴에서 128x128로 저장했지만 안전장치)
        gray = cv2.resize(gray, (128, 128))

        # 3. HOG 특징 추출
        # feature_vector는 1차원 배열(vector) 형태로 반환됩니다.
        fd = hog(gray, **HOG_PARAMS)
        
        features.append(fd)
        labels.append(label)

    return features, labels

if __name__ == "__main__":
    print(">>> HOG 특징 추출 시작")

    # Positive 데이터 (Label: 1)
    pos_features, pos_labels = load_images_and_extract_hog(POS_DATA_DIR, 1)
    
    # Negative 데이터 (Label: 0)
    neg_features, neg_labels = load_images_and_extract_hog(NEG_DATA_DIR, 0)

    # 데이터 합치기
    X = np.array(pos_features + neg_features)
    y = np.array(pos_labels + neg_labels)

    print("------------------------------------------------")
    print(f"전체 데이터 개수: {len(X)}")
    print(f"특징 벡터의 길이(차원): {X.shape[1]}") # (128/8 - 1) ... 계산 결과

    # 데이터 저장 (joblib 사용이 pickle보다 대용량 numpy 배열에 효율적)
    data_to_save = {'X': X, 'y': y}
    joblib.dump(data_to_save, OUTPUT_FILE)
    
    print(f"데이터 저장 완료: {OUTPUT_FILE}")
    print("이제 학습 코드(train_svm.py)를 실행할 수 있습니다.")