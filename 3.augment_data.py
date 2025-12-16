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

# 증강 배수 (원본 포함 몇 배로 늘릴지)
# 여기서는 원본 1장 + 증강 2장 = 총 3배
AUGMENT_COUNT = 2 

def rotate_image(image):
    """이미지를 랜덤 각도(-15 ~ +15)로 회전시킵니다."""
    angle = random.randint(-15, 15)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 회전 변환 행렬 구하기
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 이미지 회전 (빈 공간은 가장자리 색으로 채움)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated

def change_brightness(image):
    """이미지의 밝기를 랜덤하게 조절합니다."""
    value = random.randint(-50, 50)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value >= 0:
        v = cv2.add(v, value)
    else:
        v = cv2.subtract(v, abs(value))

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def add_noise(image):
    """가우시안 노이즈 추가 (화질 저하 시뮬레이션)"""
    row, col, ch = image.shape
    mean = 0
    sigma = 25  # 노이즈 강도
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    
    # 값이 0~255를 벗어나지 않도록 클리핑
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def add_blur(image):
    """모션 블러 추가 (움직임 시뮬레이션)"""
    size = random.choice([3, 5, 7]) # 커널 크기 (홀수)
    
    # 일반 가우시안 블러 사용 (혹은 Motion Blur 커널 직접 구현 가능)
    # 여기서는 간단히 GaussianBlur 사용
    blurred = cv2.GaussianBlur(image, (size, size), 0)
    return blurred

def horizontal_flip(image):
    """좌우 반전"""
    return cv2.flip(image, 1)

def augment_and_save(src_folder, dst_folder, prefix):
    # 폴더 생성
    os.makedirs(dst_folder, exist_ok=True)
    
    # 이미지 파일 목록
    files = glob.glob(os.path.join(src_folder, "*.jpg")) + \
            glob.glob(os.path.join(src_folder, "*.png"))
    
    print(f"Processing {src_folder} -> {len(files)} images found.")
    
    count = 0
    
    # 사용 가능한 증강 함수 리스트
    aug_functions = [
        (horizontal_flip, "flip"),
        (rotate_image, "rot"),
        (change_brightness, "brt"),
        (add_noise, "noise"),
        (add_blur, "blur")
    ]
    
    for filepath in files:
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        
        img = cv2.imread(filepath)
        if img is None: continue

        # 1. 원본 저장 (필수)
        cv2.imwrite(os.path.join(dst_folder, f"{name}_orig{ext}"), img)
        count += 1
        
        # 2. 랜덤 증강 적용 (AUGMENT_COUNT 만큼 선택)
        # 중복 없이 선택하기 위해 sample 사용
        selected_augs = random.sample(aug_functions, AUGMENT_COUNT)
        
        for i, (func, _) in enumerate(selected_augs):
            # 증강 적용
            aug_img = func(img)
            
            # 파일 저장 (증강 종류를 숨기고 인덱스만 사용)
            save_name = f"{name}_aug{i}{ext}"
            cv2.imwrite(os.path.join(dst_folder, save_name), aug_img)
            count += 1

    print(f"Finished! Total {count} images saved to {dst_folder}")
    print(f"(Factor: {count / len(files):.1f}x)")

if __name__ == "__main__":
    print(">>> 데이터 증강 시작... (Target: 3x Size)")
    
    # Positive 데이터 증강
    augment_and_save(SRC_POS_DIR, DST_POS_DIR, "pos")
    
    # Negative 데이터 증강
    augment_and_save(SRC_NEG_DIR, DST_NEG_DIR, "neg")
    
    print(">>> 모든 작업 완료.")
    print(f"확인하러 가기 -> Positive: {DST_POS_DIR}, Negative: {DST_NEG_DIR}")