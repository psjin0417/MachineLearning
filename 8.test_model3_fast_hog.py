import cv2
import joblib
import numpy as np
import os
import pickle
from skimage.feature import hog

# --- 설정 변수 ---
VIDEO_PATH = "./video/8.mp4"       
MODEL_PATH = "./svm_model_v3_gridsearch.pkl"          
ROI_FILE = "./roi_config.pkl"   # ROI 설정 저장 파일
TARGET_SIZE = (128, 128)           

# 리사이즈 및 회전 설정
RESIZE_SCALE = 0.4            
ROTATE_CODE = None 
SHOW_HEATMAP = True            # True: 히트맵 보이기, False: 바운딩박스만 보이기 

# [중요] 박스 예측 정밀도 조절 파라미터
HEATMAP_THRESH = 150            # (0~255) 높을수록 박스가 작고 확실해짐
CONFIDENCE_THRESHOLD = 0.95     # (0.0~1.0) 모델 확신도 임계값

# [중요] 피라미드 스케일 설정
SCALE_STEP = 1.1
MIN_SIZE = (128, 128)

# HOG & Color 파라미터
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (16, 16),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'visualize': False,
    'transform_sqrt': True,
    'feature_vector': False 
}
HIST_BINS = (32, 32)
PIXELS_PER_CELL = 16

# --- 전역 변수 (마우스 이벤트용) ---
drawing_points = []

def mouse_callback(event, x, y, flags, param):
    global drawing_points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_points.append((x, y))
        print(f"Point added: ({x}, {y})")

def get_polygon_roi(window_name, frame, message):
    """사용자가 점을 찍어 다각형 ROI를 만들도록 함"""
    global drawing_points
    drawing_points = []
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        display_frame = frame.copy()
        
        # 안내 문구
        cv2.putText(display_frame, message, (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(display_frame, "Click points. Press 'n' to finish, 'r' to reset.", (30, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 찍은 점들 이어 그리기
        if len(drawing_points) > 0:
            pts = np.array(drawing_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
            
            # 각 점 표시
            for pt in drawing_points:
                cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('n'): # Next
            if len(drawing_points) < 3:
                print("최소 3개의 점이 필요합니다.")
                continue
            break
        elif key == ord('r'): # Reset
            drawing_points = []
            
    cv2.destroyWindow(window_name)
    return np.array(drawing_points, dtype=np.int32)

def extract_color_histogram(image, bins=(32, 32)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, bins, [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def detect_multi_scale(model, frame, roi_mask=None):
    heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
    img_tosearch = frame.copy()
    
    if RESIZE_SCALE != 1.0:
        width = int(img_tosearch.shape[1] * RESIZE_SCALE)
        height = int(img_tosearch.shape[0] * RESIZE_SCALE)
        img_tosearch = cv2.resize(img_tosearch, (width, height), interpolation=cv2.INTER_AREA)

    scale = 1.0 
    
    # [최적화] 배치 처리를 위한 리스트
    batch_features = []
    batch_coords = [] # (real_x, real_y, real_w, real_h)
    
    while True:
        if img_tosearch.shape[0] < TARGET_SIZE[1] or img_tosearch.shape[1] < TARGET_SIZE[0]:
            break
            
        gray = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2GRAY)
        # HOG Feature Map 미리 계산 (한 번만!)
        hog_features_map = hog(gray, **HOG_PARAMS)
        
        n_blocks_per_window = (TARGET_SIZE[0] // PIXELS_PER_CELL) - 1
        step_cells = 1 
        
        max_y_cells = hog_features_map.shape[0] - n_blocks_per_window
        max_x_cells = hog_features_map.shape[1] - n_blocks_per_window
        
        if max_y_cells < 0 or max_x_cells < 0:
            break

        # 현재 스케일에서의 좌표 계산 상수를 미리 계산
        curr_scale_inv = 1.0 / (RESIZE_SCALE / scale)
        real_w = int(TARGET_SIZE[0] * curr_scale_inv)
        real_h = int(TARGET_SIZE[1] * curr_scale_inv)

        for yc in range(0, max_y_cells, step_cells):
            for xc in range(0, max_x_cells, step_cells):
                
                # 1. HOG Feature 잘라오기 (슬라이싱) - 매우 빠름
                feature_chunk = hog_features_map[yc : yc + n_blocks_per_window, 
                                               xc : xc + n_blocks_per_window, :, :, :]
                hog_feat = feature_chunk.ravel() 
                
                if len(hog_feat) != 1764: 
                    continue

                x_pixel = xc * PIXELS_PER_CELL
                y_pixel = yc * PIXELS_PER_CELL
                
                # 원본 좌표 계산
                real_x = int(x_pixel * curr_scale_inv)
                real_y = int(y_pixel * curr_scale_inv)
                center_x = real_x + real_w // 2
                center_y = real_y + real_h // 2
                
                # [최적화] ROI 밖이면 특징 추출도 하지 않고 스킵 (Feature Extraction 부하 감소)
                if roi_mask is not None:
                     # 이미지 범위 체크
                     if not (0 <= center_y < roi_mask.shape[0] and 0 <= center_x < roi_mask.shape[1]):
                         continue
                     # ROI 마스크 체크
                     if roi_mask[center_y, center_x] == 0:
                         continue

                # 2. Color Feature 추출 (이미지 crop 필요) - 여기가 병목지점
                window_img = img_tosearch[y_pixel : y_pixel + TARGET_SIZE[1], 
                                          x_pixel : x_pixel + TARGET_SIZE[0]]
                
                if window_img.shape[0] != TARGET_SIZE[1] or window_img.shape[1] != TARGET_SIZE[0]:
                    continue
                    
                color_feat = extract_color_histogram(window_img, bins=HIST_BINS)
                
                # 3. 결합 및 배치 추가
                combined = np.hstack([hog_feat, color_feat])
                batch_features.append(combined)
                batch_coords.append((real_x, real_y, real_w, real_h))

        img_tosearch = cv2.resize(img_tosearch, 
                                (int(img_tosearch.shape[1] / SCALE_STEP), 
                                 int(img_tosearch.shape[0] / SCALE_STEP)))
        scale *= SCALE_STEP

    # [핵심] 배치 예측 (Vectorized Prediction) - 한 번에 수천 개 예측 수행
    if len(batch_features) > 0:
        # numpy array로 변환 (매우 중요)
        X_batch = np.array(batch_features)
        
        # 한 번에 확률 계산
        all_probs = model.predict_proba(X_batch) # shape: (N, 2)
        
        # 결과 히트맵에 적용
        for i, prob_pair in enumerate(all_probs):
            prob = prob_pair[1] # Positive 확률
            
            if prob > CONFIDENCE_THRESHOLD:
                rx, ry, rw, rh = batch_coords[i]
                
                real_y2 = min(ry + rh, frame.shape[0])
                real_x2 = min(rx + rw, frame.shape[1])
                
                heatmap[ry:real_y2, rx:real_x2] += prob

    return heatmap

def load_or_create_roi(frame):
    if os.path.exists(ROI_FILE):
        try:
            with open(ROI_FILE, 'rb') as f:
                rois = pickle.load(f)
            print(">>> 기존 ROI 설정을 불러왔습니다.")
            return rois['detection'], rois['danger']
        except Exception as e:
            print(f"Error loading ROI: {e}")
    
    # 설정 파일이 없거나 에러나면 새로 생성
    print(">>> ROI 설정을 시작합니다.")
    detection_roi = get_polygon_roi("Set Detection ROI", frame, "1. Draw Detection ROI (Green)")
    danger_roi = get_polygon_roi("Set Danger Zone", frame, "2. Draw Danger/Warning Zone (ROI inside)")
    
    # 저장
    try:
        with open(ROI_FILE, 'wb') as f:
            pickle.dump({'detection': detection_roi, 'danger': danger_roi}, f)
        print(">>> ROI 설정이 저장되었습니다.")
    except Exception as e:
        print(f"Error saving ROI: {e}")
        
    return detection_roi, danger_roi

if __name__ == "__main__":
    print(">>> 모델 로딩 중...")
    model = joblib.load(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: 동영상을 열 수 없습니다.")
        exit()
        
    ret, first_frame = cap.read()
    if not ret:
        print("Error: 첫 프레임을 읽을 수 없습니다.")
        exit()
    
    if ROTATE_CODE is not None:
        first_frame = cv2.rotate(first_frame, ROTATE_CODE)

    # --- [Step 1] ROI 로드 또는 생성 ---
    detection_roi_poly, danger_roi_poly = load_or_create_roi(first_frame)
    
    # ROI Mask 생성 (Detection ROI용)
    roi_mask = np.zeros(first_frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [detection_roi_poly], 255) 

    # 비디오 다시 처음부터
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if ROTATE_CODE is not None:
            frame = cv2.rotate(frame, ROTATE_CODE)

        heatmap = detect_multi_scale(model, frame, roi_mask)
        
        # 후처리
        heatmap_norm = np.clip(heatmap, 0, 255) 
        if np.max(heatmap_norm) > 0:
            heatmap_norm = heatmap_norm / np.max(heatmap_norm) * 255
        heatmap_norm = heatmap_norm.astype(np.uint8)

        _, thresh = cv2.threshold(heatmap_norm, HEATMAP_THRESH, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = frame.copy() 
        
        # --- ROI 시각화 (반투명 오버레이) ---
        overlay = result.copy()
        # 1. Detection Zone (아주 연한 파랑)
        cv2.fillPoly(overlay, [detection_roi_poly], (255, 255, 0)) # BGR
        # 2. Danger Zone (아주 연한 빨강)
        cv2.fillPoly(overlay, [danger_roi_poly], (0, 0, 255))
        
        # 투명도 적용 (원본 0.8 + 오버레이 0.2)
        result = cv2.addWeighted(result, 0.8, overlay, 0.2, 0)

        # 외곽선은 진하게 그리기
        cv2.polylines(result, [detection_roi_poly], True, (255, 255, 0), 2)
        cv2.polylines(result, [danger_roi_poly], True, (0, 0, 255), 2)

        if contours:
            max_cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_cnt)
            cx = x + w // 2
            cy = y + h // 2
            
            color = (0, 255, 0) 
            status_text = "Detected"
            
            if cv2.pointPolygonTest(danger_roi_poly, (cx, cy), False) >= 0:
                color = (0, 0, 255) 
                status_text = "WARNING!"
            
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result, status_text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if SHOW_HEATMAP:
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            heatmap_color = cv2.bitwise_and(heatmap_color, heatmap_color, mask=roi_mask)
            result = cv2.addWeighted(result, 0.6, heatmap_color, 0.4, 0)

        cv2.imshow("Smart Kickboard Detector", result)
        
        # 키보드 조작: 'q' 종료, 'r' ROI 재설정
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # 파일 삭제 후 재시작 (또는 함수 호출)
            if os.path.exists(ROI_FILE):
                os.remove(ROI_FILE)
            print(">>> ROI 설정 파일이 삭제되었습니다. 프로그램을 재시작하면 다시 설정할 수 있습니다.")
            break

    cap.release()
    cv2.destroyAllWindows()
