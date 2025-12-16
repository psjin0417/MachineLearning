import cv2
import joblib
import numpy as np
from skimage.feature import hog

# --- 설정 변수 ---
VIDEO_PATH = "./video/4.mp4"       
MODEL_PATH = "./svm_model_v3_gridsearch.pkl"     
TARGET_SIZE = (128, 128)           

# [설정 1] 탐지할 윈도우 크기 목록 (거리에 따른 물체 크기 대응)
WINDOW_SIZES = [

    (128, 128), 
    (160, 160)
]

# 리사이즈 및 회전 설정
RESIZE_SCALE = 0.5              # 속도를 위해 작게 설정
ROTATE_CODE = None 
SHOW_HEATMAP = True            # True: 히트맵 보이기, False: 바운딩박스만 보이기 

# HOG 파라미터 (학습과 동일하게)
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (16, 16), # 차원을 줄이기 위해 16x16 추천
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'visualize': False,
    'transform_sqrt': True
}

# 컬러 히스토그램 파라미터 (학습과 동일하게)
HIST_BINS = (32, 32)

def extract_color_histogram(image, bins=(32, 32)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, bins, [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_features_single_image(img):
    img_resized = cv2.resize(img, TARGET_SIZE)
    
    # 1. HOG 추출
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_feature = hog(gray, **HOG_PARAMS)
    
    # 2. Color Hist 추출
    color_feature = extract_color_histogram(img_resized, bins=HIST_BINS)
    
    # 3. 결합
    combined = np.hstack([hog_feature, color_feature])
    
    return combined.reshape(1, -1)

def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

if __name__ == "__main__":
    print(">>> 모델 로딩 중...")
    model = joblib.load(MODEL_PATH)
    print(">>> 히트맵 (멀티 스케일) 예측 시작!")

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # 히트맵을 부드럽게 만들기 위해 윈도우 간격을 촘촘하게 설정 (기본값)
    BASIC_STEP_SIZE = 16 
    
    # 확률이 이 값 이상일 때만 히트맵에 열기를 더함 (노이즈 제거)
    MIN_PROB = 0.9

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. 전처리 (회전, 리사이즈)
        if ROTATE_CODE is not None:
            frame = cv2.rotate(frame, ROTATE_CODE)

        if RESIZE_SCALE != 1.0:
            width = int(frame.shape[1] * RESIZE_SCALE)
            height = int(frame.shape[0] * RESIZE_SCALE)
            
            if width < TARGET_SIZE[0] or height < TARGET_SIZE[1]:
                print("이미지가 너무 작습니다.")
                break
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # -----------------------------------------------------------
        # [핵심] 히트맵 초기화 (0으로 채워진 흑백 도화지 생성)
        # -----------------------------------------------------------
        heatmap = np.zeros(frame.shape[:2], dtype=np.float32)

        # 멀티 스케일 슬라이딩 윈도우 수행
        for win_size in WINDOW_SIZES:
            # 윈도우 크기에 비례하여 보폭 조절 (예: 윈도우의 1/8)
            current_step = int(max(16, win_size[0] / 8))

            for (x, y, window) in sliding_window(frame, step_size=current_step, window_size=win_size):
                if window.shape[0] != win_size[1] or window.shape[1] != win_size[0]:
                    continue
                
                # 모델 입력 크기로 리사이즈하여 특징 추출
                features = extract_features_single_image(window)
                
                # [0][1]은 Positive(킥보드)일 확률
                prob = model.predict_proba(features)[0][1]
                
                # 일정 확률 이상일 때만 히트맵에 점수 누적
                if prob > MIN_PROB:
                    # 해당 윈도우 영역에 확률 값을 더해줌
                    # 윈도우 크기가 다르므로, 큰 윈도우는 넓은 영역에 점수를 줌
                    heatmap[y:y+win_size[1], x:x+win_size[0]] += prob

        # -----------------------------------------------------------
        # [시각화] 히트맵을 컬러로 변환하여 원본에 덮어씌우기
        # -----------------------------------------------------------
        
        # -----------------------------------------------------------
        # [후처리] 히트맵 분석 및 바운딩 박스 추출
        # -----------------------------------------------------------
        
        # 1. 정규화 (0 ~ 255)
        heatmap_norm = np.clip(heatmap, 0, 255) 
        if np.max(heatmap_norm) > 0:
            heatmap_norm = heatmap_norm / np.max(heatmap_norm) * 255
        heatmap_norm = heatmap_norm.astype(np.uint8)

        # 2. 하나의 바운딩 박스 추출
        # 히트맵에서 127(약 50% 강도) 이상인 영역만 찾음
        _, thresh = cv2.threshold(heatmap_norm, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 가장 큰 영역 하나만 선택
            max_cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_cnt)
            
            # 바운딩 박스 그리기 (초록색)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Kickboard", (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # -----------------------------------------------------------
        # [시각화] 옵션에 따른 출력
        # -----------------------------------------------------------
        if SHOW_HEATMAP:
            # 컬러맵 적용 (JET)
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            # 원본 + 히트맵 합성
            result = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
        else:
            # 바운딩 박스만 그려진 원본
            result = frame

        # 결과 출력
        cv2.imshow("Kickboard Multi-Scale Heatmap", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
