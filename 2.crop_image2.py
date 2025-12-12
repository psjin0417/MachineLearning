import cv2
import os
import glob
from datetime import datetime  # [추가] 시간 정보를 위해 import

# --- 설정 변수 ---
RAW_DATA_DIR = "./raw_data/yh/3"     # 원본 이미지가 있는 폴더
POS_DATA_DIR = "./positive"     # Positive 데이터 저장 폴더
NEG_DATA_DIR = "./negative"     # Negative 데이터 저장 폴더
TARGET_SIZE = (128, 128)        # 저장할 이미지 크기 (128 x 128)

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
        self.img = None
        self.temp_img = None
        
        # 파일명 카운터 (폴더 내 파일 개수로 초기화하지만, 시간 정보가 있어 중복 걱정 없음)
        self.pos_count = len(os.listdir(POS_DATA_DIR))
        self.neg_count = len(os.listdir(NEG_DATA_DIR))

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 이벤트 처리 함수: 드래그로 영역 지정"""
        if event == cv2.EVENT_LBUTTONDOWN: # 클릭 시작
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE: # 드래그 중
            if self.drawing:
                self.temp_img = self.img.copy()
                # 모드에 따라 박스 색상 변경 (Positive: 초록, Negative: 빨강)
                color = (0, 255, 0) if self.mode == "positive" else (0, 0, 255)
                cv2.rectangle(self.temp_img, (self.ix, self.iy), (x, y), color, 2)
                cv2.imshow("Labeling Tool", self.temp_img)

        elif event == cv2.EVENT_LBUTTONUP: # 클릭 해제 (크롭 및 저장)
            self.drawing = False
            self.save_crop(x, y)
            # 박스가 그려진 상태를 원본 이미지로 업데이트하지 않고, 다음 크롭을 위해 원본 유지
            cv2.imshow("Labeling Tool", self.img)

    def save_crop(self, x, y):
        """선택된 영역을 잘라내고 리사이징 후 저장"""
        # 좌표 정렬 (역방향 드래그 대응)
        x1, x2 = min(self.ix, x), max(self.ix, x)
        y1, y2 = min(self.iy, y), max(self.iy, y)
        
        w, h = x2 - x1, y2 - y1

        # 너무 작은 영역(실수)은 무시
        if w < 10 or h < 10:
            print("영역이 너무 작습니다. 저장을 건너뜁니다.")
            return

        # 이미지 크롭
        roi = self.img[y1:y2, x1:x2]
        
        if roi.size == 0:
            return

        # 128x128로 리사이징
        resized_roi = cv2.resize(roi, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

        # [수정] 현재 시간 가져오기
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S") # 예: 20241124_150530

        # 저장 경로 및 파일명 설정 (시간_카운트 조합)
        if self.mode == "positive":
            filename = f"pos_{timestamp}_{self.pos_count:05d}.jpg"
            save_path = os.path.join(POS_DATA_DIR, filename)
            self.pos_count += 1
            print(f"[Positive] 저장 완료: {filename}")
        else:
            filename = f"neg_{timestamp}_{self.neg_count:05d}.jpg"
            save_path = os.path.join(NEG_DATA_DIR, filename)
            self.neg_count += 1
            print(f"[Negative] 저장 완료: {filename}")

        # 이미지 저장
        cv2.imwrite(save_path, resized_roi)

    def run(self):
        # [수정 1] 윈도우 생성 시 WINDOW_NORMAL 플래그 사용 (크기 조절 가능 모드)
        cv2.namedWindow("Labeling Tool", cv2.WINDOW_NORMAL)
        
        # [수정 2] 윈도우 크기를 강제로 지정 (예: 1280x960, 본인 모니터에 맞게 조절)
        cv2.resizeWindow("Labeling Tool", 1280, 960) 

        cv2.setMouseCallback("Labeling Tool", self.mouse_callback)

        print("=== 라벨링 툴 시작 ===")
        print("[M] 키: 모드 전환 (Positive <-> Negative)")
        print("[Space] 키: 다음 이미지로 이동")
        print("[ESC] 키: 종료")
        print("마우스 드래그: 영역 선택 및 자동 저장")

        while True:
            if self.current_idx >= len(self.image_paths):
                print("모든 이미지를 확인했습니다.")
                break

            image_path = self.image_paths[self.current_idx]
            self.img = cv2.imread(image_path)
            
            if self.img is None:
                self.current_idx += 1
                continue

            self.temp_img = self.img.copy()

            while True:
                # 화면에 현재 모드 및 정보 텍스트 출력
                display_img = self.temp_img.copy() if self.drawing else self.img.copy()
                
                status_color = (0, 255, 0) if self.mode == "positive" else (0, 0, 255)
                cv2.putText(display_img, f"Mode: {self.mode.upper()}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
                # 파일명 출력
                img_name = os.path.basename(image_path)
                cv2.putText(display_img, f"Image: {img_name} ({self.current_idx+1}/{len(self.image_paths)})", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow("Labeling Tool", display_img)
                
                key = cv2.waitKey(1) & 0xFF

                # [ESC] 종료
                if key == 27: 
                    print("프로그램을 종료합니다.")
                    cv2.destroyAllWindows()
                    return
                
                # [Space] 다음 이미지
                elif key == ord(' '):
                    self.current_idx += 1
                    print(">>> 다음 이미지로 이동")
                    break # 내부 while 탈출 -> 외부 while에서 다음 이미지 로드
                
                # [M] 모드 전환
                elif key == ord('m'):
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