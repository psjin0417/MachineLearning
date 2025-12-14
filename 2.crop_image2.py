import cv2
import os
import glob
from datetime import datetime  # [추가] 시간 정보를 위해 import

# --- 설정 변수 ---
RAW_DATA_DIR = "./raw_data"     # 원본 이미지가 있는 폴더
POS_DATA_DIR = "./positive"     # Positive 데이터 저장 폴더
NEG_DATA_DIR = "./negative"     # Negative 데이터 저장 폴더
TARGET_SIZE = (128, 128)        # 저장할 이미지 크기 (128 x 128)
DISPLAY_SCALE = 1.5            # [추가] 화면 표시 배율 (예: 2.0 -> 2배 확대 표시)

class DataLabeler:
    def __init__(self):
        # 디렉토리 생성
        os.makedirs(POS_DATA_DIR, exist_ok=True)                      
        os.makedirs(NEG_DATA_DIR, exist_ok=True)

        # 이미지 리스트 불러오기 (jpg, png 등)
        self.image_paths = glob.glob(os.path.join(RAW_DATA_DIR, "*.jpg")) + \
                           glob.glob(os.path.join(RAW_DATA_DIR, "*.png"))
        self.image_paths.sort()
        
        if not self.image_paths:
            print(f"Error: '{RAW_DATA_DIR}' 폴더에 이미지가 없습니다.")
            exit()

        # 상태 변수 초기화
        self.mode = "positive"  # 초기 모드: positive
        self.current_idx = 0
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.original_img = None    # [수정] 원본 이미지
        self.display_img = None     # [추가] 화면 표시용 확대 이미지
        self.temp_display_img = None 
        
        # 파일명 카운터
        self.pos_count = len(os.listdir(POS_DATA_DIR))
        self.neg_count = len(os.listdir(NEG_DATA_DIR))

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 이벤트 처리 함수: 드래그로 영역 지정"""
        # 화면 좌표 -> 원본 좌표 변환
        real_x = int(x / DISPLAY_SCALE)
        real_y = int(y / DISPLAY_SCALE)

        if event == cv2.EVENT_LBUTTONDOWN: # 클릭 시작
            self.drawing = True
            self.ix, self.iy = x, y # 드로잉은 화면 좌표계 사용
            self.real_ix, self.real_iy = real_x, real_y # 크롭은 원본 좌표계 저장

        elif event == cv2.EVENT_MOUSEMOVE: # 드래그 중
            if self.drawing:
                self.temp_display_img = self.display_img.copy()
                # 모드에 따라 박스 색상 변경
                color = (0, 255, 0) if self.mode == "positive" else (0, 0, 255)
                
                # 화면에 보여지는 박스 그리기 (화면 좌표 사용)
                cv2.rectangle(self.temp_display_img, (self.ix, self.iy), (x, y), color, 2)
                
                # 텍스트 정보 표시 (옵션)
                w = abs(real_x - self.real_ix)
                h = abs(real_y - self.real_iy)
                cv2.putText(self.temp_display_img, f"{w}x{h}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                cv2.imshow("Labeling Tool", self.temp_display_img)

        elif event == cv2.EVENT_LBUTTONUP: # 클릭 해제 (크롭 및 저장)
            self.drawing = False
            # 크롭은 원본 좌표계를 기준으로 수행해야 함
            self.save_crop(real_x, real_y)
            
            # 박스 그려진 것 초기화 (원본 상태로 복귀)
            cv2.imshow("Labeling Tool", self.display_img)

    def save_crop(self, x, y):
        """선택된 영역을 잘라내고 리사이징 후 저장 (원본 좌표계 x, y)"""
        # 좌표 정렬
        x1, x2 = min(self.real_ix, x), max(self.real_ix, x)
        y1, y2 = min(self.real_iy, y), max(self.real_iy, y)
        
        # 원본 이미지 범위 체크
        h_img, w_img = self.original_img.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_img, x2); y2 = min(h_img, y2)

        w, h = x2 - x1, y2 - y1

        # 너무 작은 영역 무시
        if w < 10 or h < 10:
            print("영역이 너무 작습니다. 저장을 건너뜁니다.")
            return

        # 이미지 크롭 (원본 이미지에서)
        roi = self.original_img[y1:y2, x1:x2]
        
        if roi.size == 0:
            return

        # 128x128로 리사이징
        try:
            resized_roi = cv2.resize(roi, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"Error resize: {e}")
            return

        # 저장 로직
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        if self.mode == "positive":
            filename = f"pos_{timestamp}_{self.pos_count:05d}.jpg"
            save_path = os.path.join(POS_DATA_DIR, filename)
            self.pos_count += 1
            print(f"[Positive] 저장 완료: {filename} (Src: {w}x{h})")
        else:
            filename = f"neg_{timestamp}_{self.neg_count:05d}.jpg"
            save_path = os.path.join(NEG_DATA_DIR, filename)
            self.neg_count += 1
            print(f"[Negative] 저장 완료: {filename} (Src: {w}x{h})")

        cv2.imwrite(save_path, resized_roi)

    def run(self):
        cv2.namedWindow("Labeling Tool", cv2.WINDOW_NORMAL)
        # 화면 크기 설정은 이미지가 로드된 후 resizeWindow 등을 할 수도 있지만, 
        # 여기서는 단순히 창을 띄웁니다.
        
        print("=== 라벨링 툴 시작 ===")
        print(f"DISPLAY_SCALE: {DISPLAY_SCALE} (화면 확대 배율)")
        print("[M] 키: 모드 전환 (Positive <-> Negative)")
        print("[Space] 키: 다음 이미지로 이동")
        print("[ESC] 키: 종료")
        print("마우스 드래그: 영역 선택 및 자동 저장")

        cv2.setMouseCallback("Labeling Tool", self.mouse_callback)

        while True:
            if self.current_idx >= len(self.image_paths):
                print("모든 이미지를 확인했습니다.")
                break

            image_path = self.image_paths[self.current_idx]
            self.original_img = cv2.imread(image_path) # 원본 로드
            
            if self.original_img is None:
                self.current_idx += 1
                continue
                
            # [추가] 표시용 이미지 생성 (확대)
            h, w = self.original_img.shape[:2]
            new_w = int(w * DISPLAY_SCALE)
            new_h = int(h * DISPLAY_SCALE)
            self.display_img = cv2.resize(self.original_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            self.temp_display_img = self.display_img.copy()

            # 윈도우 크기를 확대된 이미지에 맞추거나 적절히 조절
            # cv2.resizeWindow("Labeling Tool", new_w, new_h) 

            while True:
                # 화면에 그릴 대상은 temp_display_img (드래그 중일때) 혹은 display_img (평시)
                # 다만 여기 구조상 while 루프 돌때마다 텍스트를 새로 박아야 하므로 copy 사용
                if self.drawing:
                    show_img = self.temp_display_img # 마우스 콜백에서 그려짐
                else:
                    show_img = self.display_img.copy()
                    
                    # 상태 텍스트
                    status_color = (0, 255, 0) if self.mode == "positive" else (0, 0, 255)
                    cv2.putText(show_img, f"Mode: {self.mode.upper()}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                    
                    img_name = os.path.basename(image_path)
                    #cv2.putText(show_img, f"Img: {img_name} ({self.current_idx+1}/{len(self.image_paths)})", 
                    ##            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Labeling Tool", show_img)
                
                key = cv2.waitKey(1) & 0xFF

                if key == 27: # ESC
                    print("프로그램을 종료합니다.")
                    cv2.destroyAllWindows()
                    return
                
                elif key == ord(' '): # Space
                    self.current_idx += 1
                    print(">>> 다음 이미지로 이동")
                    break
                
                elif key == ord('m'): # M
                    if self.mode == "positive":
                        self.mode = "negative"
                    else:
                        self.mode = "positive"
                    print(f"모드 변경됨: {self.mode.upper()}")

        cv2.destroyAllWindows()

# 실행
if __name__ == "__main__":
    labeler = DataLabeler()
    labeler.run()