import cv2
import os
import numpy as np
import glob
import random

# --- 설정 변수 ---
# 원본 데이터 경로
SRC_POS_DIR = "./positive"
SRC_NEG_DIR = "./negative"

# 증강된 데이터가 저장될 새로운 경로
DST_POS_DIR = "./positive_aug"
DST_NEG_DIR = "./negative_aug"

def rotate_image(image, angle):
    """이미지를 중심 기준으로 회전시킵니다."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 회전 변환 행렬 구하기
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 이미지 회전 (빈 공간은 가장자리 색으로 채움)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated

def change_brightness(image):
    """이미지의 밝기를 랜덤하게 조절합니다."""
    # -50 ~ +50 사이의 값을 랜덤으로 더함
    value = random.randint(-50, 50)
    
    # HSV로 변환하여 V(명도) 채널만 조절하는 것이 색상 왜곡이 적음
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # cv2.add는 255를 넘거나 0 밑으로 가면 알아서 클리핑(clipping)해줌
    if value >= 0:
        v = cv2.add(v, value)
    else:
        v = cv2.subtract(v, abs(value))

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def augment_and_save(src_folder, dst_folder, prefix):
    # 폴더 생성
    os.makedirs(dst_folder, exist_ok=True)
    
    # 이미지 파일 목록
    files = glob.glob(os.path.join(src_folder, "*.jpg")) + \
            glob.glob(os.path.join(src_folder, "*.png"))
    
    print(f"Processing {src_folder} -> {len(files)} images found.")
    
    count = 0
    
    for filepath in files:
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        
        img = cv2.imread(filepath)
        if img is None: continue

        # ---------------------------------------------------------
        # 1. 원본 저장 (Original)
        # ---------------------------------------------------------
        cv2.imwrite(os.path.join(dst_folder, f"{name}_orig{ext}"), img)
        count += 1

        # ---------------------------------------------------------
        # 2. 수평 뒤집기 (Horizontal Flip) - 좌우 반전
        # ---------------------------------------------------------
        # 킥보드가 왼쪽을 보나 오른쪽을 보나 킥보드임
        flipped = cv2.flip(img, 1) 
        cv2.imwrite(os.path.join(dst_folder, f"{name}_flip{ext}"), flipped)
        count += 1

        # ---------------------------------------------------------
        # 3. 회전 (Rotation) - 랜덤 각도 (-15도 ~ +15도)
        # ---------------------------------------------------------
        # 약간 기울어진 킥보드도 인식하기 위해
        angle = random.randint(-15, 15)
        rotated = rotate_image(img, angle)
        cv2.imwrite(os.path.join(dst_folder, f"{name}_rot{ext}"), rotated)
        count += 1

        # ---------------------------------------------------------
        # 4. 밝기 조절 (Brightness) - 랜덤 밝기
        # ---------------------------------------------------------
        # 야외 조명 환경 대응 (추천)
        bright_img = change_brightness(img)
        cv2.imwrite(os.path.join(dst_folder, f"{name}_brt{ext}"), bright_img)
        count += 1

    print(f"Finished! Total {count} images saved to {dst_folder}")

if __name__ == "__main__":
    print(">>> 데이터 증강 시작...")
    
    # Positive 데이터 증강
    augment_and_save(SRC_POS_DIR, DST_POS_DIR, "pos")
    
    # Negative 데이터 증강
    augment_and_save(SRC_NEG_DIR, DST_NEG_DIR, "neg")
    
    print(">>> 모든 작업 완료.")
    print(f"확인하러 가기 -> Positive: {DST_POS_DIR}, Negative: {DST_NEG_DIR}")