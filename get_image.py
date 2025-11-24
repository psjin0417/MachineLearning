import cv2
import os
from datetime import datetime  # [추가] 날짜/시간 기능을 위해 필요

def save_frames_from_video(video_path, output_folder, interval=30, rotate_code=None):
    """
    동영상에서 일정 간격으로 프레임을 추출하여 저장하는 함수
    (파일명에 시간을 포함하여 중복 방지)
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

            # 2. [수정] 파일명에 현재 시간(년월일_시분초) 추가
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
    my_video = "./video/3.mp4" 
    save_dir = "./raw_data"
    
    # 회전 옵션 (필요 없으면 None)
    rotation = cv2.ROTATE_90_CLOCKWISE 
    
    # 실행
    save_frames_from_video(my_video, save_dir, interval=10, rotate_code=rotation)