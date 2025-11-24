import cv2
import joblib
import numpy as np
from skimage.feature import hog

# --- 설정 변수 ---
VIDEO_PATH = "./video/3.mp4"       
MODEL_PATH = "./svm_model.pkl"     
TARGET_SIZE = (128, 128)           

# [설정 1] 리사이즈 비율 (0.1 ~ 1.0)
# 1.0 = 원본 크기 그대로 (가장 느림, 작은 물체 탐지 유리)
# 0.5 = 절반 크기로 줄임 (속도 4배 빠름, 큰 물체 탐지 유리)
RESIZE_SCALE = 0.2

# [설정 2] 회전 설정
# cv2.ROTATE_90_CLOCKWISE: 시계방향 90도
# None: 회전 안 함
ROTATE_CODE = cv2.ROTATE_90_CLOCKWISE 

# [중요] HOG 파라미터 (학습 때와 동일하게!)
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),     # 8x8 (8100 features)
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'visualize': False,
    'transform_sqrt': True
}

def extract_features_single_image(img):
    img_resized = cv2.resize(img, TARGET_SIZE)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_feature = hog(gray, **HOG_PARAMS)
    return hog_feature.reshape(1, -1)

def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# --- 메인 실행 ---
if __name__ == "__main__":
    print(">>> 모델 로딩 중...")
    model = joblib.load(MODEL_PATH)
    print(">>> 모델 로드 완료! 탐지를 시작합니다.")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: 동영상을 열 수 없습니다.")
        exit()

    # 슬라이딩 윈도우 간격
    STEP_SIZE = 16
    CONFIDENCE_THRESHOLD = 0.6 

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. 회전 (먼저 회전시켜서 올바른 방향을 잡음)
        if ROTATE_CODE is not None:
            frame = cv2.rotate(frame, ROTATE_CODE)

        # 2. 비율 유지 리사이즈 (Aspect Ratio Preserved)
        if RESIZE_SCALE != 1.0:
            width = int(frame.shape[1] * RESIZE_SCALE)
            height = int(frame.shape[0] * RESIZE_SCALE)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        display_frame = frame.copy()
        
        # 슬라이딩 윈도우 수행
        for (x, y, window) in sliding_window(frame, step_size=STEP_SIZE, window_size=TARGET_SIZE):
            if window.shape[0] != TARGET_SIZE[1] or window.shape[1] != TARGET_SIZE[0]:
                continue
            
            features = extract_features_single_image(window)
            prob = model.predict_proba(features)[0][1]
            
            if prob > CONFIDENCE_THRESHOLD:
                # 박스 그리기
                cv2.rectangle(display_frame, (x, y), (x + TARGET_SIZE[0], y + TARGET_SIZE[1]), (0, 255, 0), 2)
                
                # 확률 텍스트
                label = f"{int(prob*100)}%"
                cv2.putText(display_frame, label, (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Kickboard Detector (Ratio Preserved)", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()