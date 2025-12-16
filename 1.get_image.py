import cv2
import os
from datetime import datetime  # [추가] 날짜/시간 기능을 위해 필요

def save_frames_from_video(video_path, output_folder, interval=30, rotate_code=None, resize_ratio=None, fixed_size=None):
    """
    동영상에서 일정 간격으로 프레임을 추출하여 저장하는 함수
    (파일명에 시간을 포함하여 중복 방지)
    resize_ratio: 이미지 크기 조절 비율 (예: 0.5 -> 절반 크기)
    fixed_size: 고정 크기 튜플 (width, height) (예: (640, 480))
                fixed_size가 설정되면 resize_ratio는 무시됩니다.
    """
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: 동영상 파일을 열 수 없습니다: {video_path}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"폴더 생성 완료: {output_folder}")

    frame_count = 0
    saved_count = 0

    print("프레임 추출을 시작합니다...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % interval == 0:
            # 1. 회전 처리
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)

            # 2. 리사이징 처리
            if fixed_size is not None:
                h, w = frame.shape[:2]
                target_w, target_h = fixed_size
                
                # 원본 비율 유지를 위한 스케일 계산
                scale = min(target_w / w, target_h / h)
                
                # 이미지가 설정한 크기보다 클 때만 줄임 (작은 이미지는 그대로 둠 혹은 필요시 늘림)
                # 여기서는 '너무 큰 이미지'를 줄이는 것이 목적이므로 scale < 1 일 때만 적용
                if scale < 1.0:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
            elif resize_ratio is not None:
                width = int(frame.shape[1] * resize_ratio)
                height = int(frame.shape[0] * resize_ratio)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            # 3. [수정] 파일명에 현재 시간(년월일_시분초) 추가
            # 포맷: YYYYMMDD_HHMMSS
            # now()가 루프 안에 있어야 미세하게라도 시간이 바뀔 수 있지만, 
            # 한 번 실행 내에서의 순서는 saved_count가 보장하므로
            # 실행 시점 구분용으로 사용합니다.
            
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S") # 예: 20231124_170530
            
            # 파일명 생성: 날짜시간_프레임번호.jpg
            image_name = f"{timestamp}_frame_{saved_count:04d}.jpg"
            image_path = os.path.join(output_folder, image_name)
            
            cv2.imwrite(image_path, frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"{saved_count}장 저장됨... ({image_name})")

        frame_count += 1

    cap.release()
    print("------------------------------------------------")
    print(f"작업 완료! 총 {saved_count}장의 이미지가 저장되었습니다.")
    print(f"저장 폴더: {output_folder}")

# --- 사용 예시 ---
if __name__ == "__main__":
    my_video = "./video/11.mp4" 
    save_dir = "./raw_data"
    
    # 회전 옵션 (필요 없으면 None)
    # rotation = cv2.ROTATE_90_CLOCKWISE 
    rotation = None
    # 실행
    # 1. 비율로 줄이기 (resize_ratio=0.5)
    # save_frames_from_video(my_video, save_dir, interval=10, rotate_code=rotation, resize_ratio=0.5)
    
    # 2. 고정 크기로 줄이기 (fixed_size=(640, 480))
    save_frames_from_video(my_video, save_dir, interval=10, rotate_code=rotation, fixed_size=None)