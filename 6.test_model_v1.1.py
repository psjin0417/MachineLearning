import cv2
import joblib
import numpy as np
from skimage.feature import hog

# --- 설정 변수 ---
VIDEO_PATH = "./video/3.mp4"       
MODEL_PATH = "./svm_model.pkl"     
TARGET_SIZE = (128, 128)           

# 리사이즈 및 회전 설정
RESIZE_SCALE = 0.3              # 속도를 위해 작게 설정
ROTATE_CODE = cv2.ROTATE_90_CLOCKWISE 

# HOG 파라미터 (학습과 동일하게)
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
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

if __name__ == "__main__":
    print(">>> 모델 로딩 중...")
    model = joblib.load(MODEL_PATH)
    print(">>> 히트맵 예측 시작!")

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # 히트맵을 부드럽게 만들기 위해 윈도우 간격을 촘촘하게 설정
    STEP_SIZE = 16 
    
    # 확률이 이 값 이상일 때만 히트맵에 열기를 더함 (노이즈 제거)
    MIN_PROB = 0.5 

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

        # 슬라이딩 윈도우 수행
        for (x, y, window) in sliding_window(frame, step_size=STEP_SIZE, window_size=TARGET_SIZE):
            if window.shape[0] != TARGET_SIZE[1] or window.shape[1] != TARGET_SIZE[0]:
                continue
            
            features = extract_features_single_image(window)
            
            # [0][1]은 Positive(킥보드)일 확률
            prob = model.predict_proba(features)[0][1]
            
            # 일정 확률 이상일 때만 히트맵에 점수 누적
            if prob > MIN_PROB:
                # 해당 윈도우 영역에 확률 값을 더해줌 (겹칠수록 값이 커짐)
                heatmap[y:y+TARGET_SIZE[1], x:x+TARGET_SIZE[0]] += prob

        # -----------------------------------------------------------
        # [시각화] 히트맵을 컬러로 변환하여 원본에 덮어씌우기
        # -----------------------------------------------------------
        
        # 1. 정규화 (0 ~ 255 사이 값으로 변환)
        # 값이 너무 작으면 안 보이니까 최댓값으로 나눠서 스케일링
        heatmap_norm = np.clip(heatmap, 0, 255) # 안전장치
        if np.max(heatmap_norm) > 0:
            heatmap_norm = heatmap_norm / np.max(heatmap_norm) * 255
        
        heatmap_norm = heatmap_norm.astype(np.uint8)

        # 2. 컬러맵 적용 (JET: 파랑(차가움) -> 빨강(뜨거움))
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        # 3. 원본 영상과 합성 (Weighted Add)
        # 원본 60% + 히트맵 40%
        result = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

        # 결과 출력
        cv2.imshow("Kickboard Heatmap", result)
        
        # 원본 히트맵(흑백)도 보고 싶으면 주석 해제
        # cv2.imshow("Heatmap Raw", heatmap_norm)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()