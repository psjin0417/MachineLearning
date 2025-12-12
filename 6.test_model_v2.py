import cv2
import joblib
import numpy as np
from skimage.feature import hog

# --- 설정 변수 ---
VIDEO_PATH = "./video/2.mp4"       
MODEL_PATH = "./svm_model.pkl"     
TARGET_SIZE = (128, 128)           

# [설정 1] 리사이즈 비율 (0.1 ~ 1.0)
RESIZE_SCALE = 0.2

# [설정 2] 회전 설정
ROTATE_CODE = cv2.ROTATE_90_CLOCKWISE 

# [설정 3] 특징 추출 모드 (V2 적용 부분)
# 'gray' : 기존 방식 (흑백)
# 'hsv_v': HSV의 V채널 사용 (조명 변화에 강함, 추천)
# 'hsv_all': HSV 모든 채널 사용 (주의: 학습된 모델의 입력 차원과 같아야 함)
FEATURE_MODE = 'hsv_v' 

# [중요] HOG 파라미터 (학습 때와 동일하게!)
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'visualize': False,
    'transform_sqrt': True,
    'channel_axis': None # 1채널(Gray/V-channel)일 경우 None
}

def extract_features_single_image(img):
    """
    이미지 한 장에서 특징을 추출하는 함수 (V2 로직 적용)
    """
    # 1. 리사이즈 (HOG 입력 크기로 맞춤)
    img_resized = cv2.resize(img, TARGET_SIZE)

    # 2. 모드에 따른 전처리 및 색상 변환
    if FEATURE_MODE == 'hsv_v':
        # [V2 핵심] RGB -> HSV 변환 후 V(명도) 채널만 추출
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        target_img = hsv[:, :, 2] # Index 2 = V Channel
        
        # 파라미터 강제 설정 (V채널은 1채널 이미지임)
        HOG_PARAMS['channel_axis'] = None 

    elif FEATURE_MODE == 'hsv_all':
        # HSV 3채널 모두 사용 (학습 모델이 이를 지원해야 함)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        target_img = hsv
        HOG_PARAMS['channel_axis'] = -1 

    else:
        # 기존 방식 (Grayscale)
        target_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        HOG_PARAMS['channel_axis'] = None

    # 3. HOG 특징 추출
    hog_feature = hog(target_img, **HOG_PARAMS)
    return hog_feature.reshape(1, -1)

def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# --- 메인 실행 ---
if __name__ == "__main__":
    print(f">>> [{FEATURE_MODE}] 모드로 모델 로딩 중...")
    
    # 모델 파일이 없으면 에러 방지
    try:
        model = joblib.load(MODEL_PATH)
        print(">>> 모델 로드 완료! 탐지를 시작합니다.")
    except FileNotFoundError:
        print(f"Error: 모델 파일({MODEL_PATH})을 찾을 수 없습니다.")
        exit()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: 동영상을 열 수 없습니다.")
        exit()

    # 슬라이딩 윈도우 간격 (작을수록 촘촘하지만 느림)
    STEP_SIZE = 16 
    CONFIDENCE_THRESHOLD = 0.6 

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. 회전
        if ROTATE_CODE is not None:
            frame = cv2.rotate(frame, ROTATE_CODE)

        # 2. 비율 유지 리사이즈
        if RESIZE_SCALE != 1.0:
            width = int(frame.shape[1] * RESIZE_SCALE)
            height = int(frame.shape[0] * RESIZE_SCALE)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        display_frame = frame.copy()
        
        # 3. 슬라이딩 윈도우 수행
        for (x, y, window) in sliding_window(frame, step_size=STEP_SIZE, window_size=TARGET_SIZE):
            if window.shape[0] != TARGET_SIZE[1] or window.shape[1] != TARGET_SIZE[0]:
                continue
            
            # 여기서 수정된 함수(V2 로직)가 호출됨
            features = extract_features_single_image(window)
            
            # 예측 수행
            prob = model.predict_proba(features)[0][1]
            
            if prob > CONFIDENCE_THRESHOLD:
                # 박스 그리기
                cv2.rectangle(display_frame, (x, y), (x + TARGET_SIZE[0], y + TARGET_SIZE[1]), (0, 255, 0), 2)
                
                # 확률 및 모드 표시
                label = f"{int(prob*100)}%"
                cv2.putText(display_frame, label, (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 화면 좌측 상단에 현재 모드 표시
        cv2.putText(display_frame, f"Mode: {FEATURE_MODE}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Kickboard Detector (V2 - HSV)", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()