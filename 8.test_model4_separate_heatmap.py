import cv2
import joblib
import numpy as np
import datetime
from skimage.feature import hog

# --- 설정 변수 ---
VIDEO_PATH = "./video/7.mp4"       
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
SHOW_HEATMAP = False             # Main Window에 히트맵 오버레이 할지 여부 (False로 변경)
SAVE_RESULT_VIDEO = True        # 결과 영상 저장 여부 (True로 재설정)

# HOG 파라미터 (학습과 동일하게)
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (16, 16),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'visualize': False,
    'transform_sqrt': True
}

# 컬러 히스토그램 파라미터 (학습과 동일하게)
HIST_BINS = (32, 32)
MIN_PROB = 0.9 # 확률 임계값

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
    if not cap.isOpened():
        print(f"Error: {VIDEO_PATH} 영상을 열 수 없습니다.")
        exit()

    # 영상 저장 설정
    video_writer = None
    if SAVE_RESULT_VIDEO:
        # 시간 포함된 파일명 생성
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        output_filename = f"result_{timestamp}.mp4"
        
        # FPS와 크기 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 리사이즈된 크기에 맞춰서 저장해야 함
        # 첫 프레임을 읽어서 실제 처리 크기를 확인해야 안전함
        ret, frame = cap.read()
        if not ret:
            print("Error: 첫 프레임을 읽을 수 없습니다.")
            exit()
            
        # 첫 프레임 기준 리사이즈 크기 계산
        if ROTATE_CODE is not None:
            frame = cv2.rotate(frame, ROTATE_CODE)
        
        if RESIZE_SCALE != 1.0:
            width = int(frame.shape[1] * RESIZE_SCALE)
            height = int(frame.shape[0] * RESIZE_SCALE)
            # [수정] 두 개의 이미지를 가로로 합칠 것이므로 너비는 2배
            frame_size = (width * 2, height)
        else:
            frame_size = (frame.shape[1] * 2, frame.shape[0])
            
        print(f">>> 녹화 시작: {output_filename} ({frame_size[0]}x{frame_size[1]})")
        # mp4v 코덱 사용
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)
        
        # 캡처 위치 초기화
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # 히트맵 스텝 사이즈
    BASIC_STEP_SIZE = 16 

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
        # [핵심] 히트맵 초기화
        # -----------------------------------------------------------
        heatmap = np.zeros(frame.shape[:2], dtype=np.float32)

        # 멀티 스케일 슬라이딩 윈도우 수행
        for win_size in WINDOW_SIZES:
            current_step = int(max(16, win_size[0] / 8))

            for (x, y, window) in sliding_window(frame, step_size=current_step, window_size=win_size):
                if window.shape[0] != win_size[1] or window.shape[1] != win_size[0]:
                    continue
                
                features = extract_features_single_image(window)
                prob = model.predict_proba(features)[0][1]
                
                if prob > MIN_PROB:
                    heatmap[y:y+win_size[1], x:x+win_size[0]] += prob

        # -----------------------------------------------------------
        # [후처리] 히트맵 분석 및 바운딩 박스 추출
        # -----------------------------------------------------------
        heatmap_norm = np.clip(heatmap, 0, 255) 
        if np.max(heatmap_norm) > 0:
            heatmap_norm = heatmap_norm / np.max(heatmap_norm) * 255
        heatmap_norm = heatmap_norm.astype(np.uint8)

        # 2. Thresholding (이진화)
        _, thresh = cv2.threshold(heatmap_norm, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = frame.copy()
        
        if contours:
            max_cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_cnt)
            
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, "Kickboard", (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # -----------------------------------------------------------
        # [시각화 1] 메인 윈도우 (Overlay 옵션) - 사용자가 끔
        # -----------------------------------------------------------
        # 사용자가 "결과창에는 히트맵 없이 바운딩 박스만" 원했으므로 
        # SHOW_HEATMAP = False로 설정했지만, 코드로도 확실히 분리
        final_view = result

        cv2.imshow("Kickboard Multi-Scale Heatmap (Result)", final_view)

        # -----------------------------------------------------------
        # [시각화 2] 별도 히트맵 윈도우 (밀집도 - 빨강/오렌지)
        # -----------------------------------------------------------
        # "밀집도(확률 합)를 색상으로 확인할 수 있게"
        # "진하게 나와야돼 (빨간색? 오렌지색)"
        # COLORMAP_HOT: Black -> Red -> Yellow -> White 순서로 강렬해짐
        # 밀집도가 높을수록 Red -> Orange/Yellow로 변함
        density_heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_HOT)
        
        cv2.imshow("Density Heatmap (Red/Orange)", density_heatmap)

        # -----------------------------------------------------------
        # [저장] 영상 녹화 (두 이미지 합치기)
        # -----------------------------------------------------------
        if video_writer is not None:
            # 두 이미지를 가로로 연결 (Horizontal Stack)
            # final_view: 결과 화면 (바운딩박스)
            # density_heatmap: 밀집도 히트맵 (빨강/오렌지)
            combined_view = np.hstack([final_view, density_heatmap])
            video_writer.write(combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f">>> 영상 저장 완료: {output_filename}")
        
    cv2.destroyAllWindows()
