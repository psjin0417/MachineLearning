import cv2
import os
import random
import glob
import numpy as np
import matplotlib.pyplot as plt

# --- 설정 변수 ---
SRC_POS_DIR = "./positive"

# --- 증강 함수들 (3.augment_data.py와 동일한 로직) ---
def rotate_image(image):
    angle = random.randint(-15, 15)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated, angle

def change_brightness(image):
    value = random.randint(-50, 50)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if value >= 0:
        v = cv2.add(v, value)
    else:
        v = cv2.subtract(v, abs(value))
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img, value

def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def add_blur(image):
    size = random.choice([3, 5, 7])
    blurred = cv2.GaussianBlur(image, (size, size), 0)
    return blurred, size

def horizontal_flip(image):
    return cv2.flip(image, 1)

def main():
    # 1. 이미지 파일 찾기
    files = glob.glob(os.path.join(SRC_POS_DIR, "*.jpg")) + \
            glob.glob(os.path.join(SRC_POS_DIR, "*.png"))
            
    if not files:
        print(f"Error: {SRC_POS_DIR}에 이미지가 없습니다.")
        return

    # 2. 랜덤 선택
    target_path = random.choice(files)
    print(f"Selected Image: {target_path}")
    
    original = cv2.imread(target_path)
    if original is None:
        print("이미지 로드 실패")
        return
        
    # BGR -> RGB 변환 (matplotlib용)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # 3. 증강 적용
    aug_results = []
    
    # (1) Flip
    flip_img = horizontal_flip(original)
    aug_results.append(("Flip", cv2.cvtColor(flip_img, cv2.COLOR_BGR2RGB)))
    
    # (2) Rotate
    rot_img, angle = rotate_image(original)
    aug_results.append((f"Rotate ({angle}deg)", cv2.cvtColor(rot_img, cv2.COLOR_BGR2RGB)))
    
    # (3) Brightness
    brt_img, val = change_brightness(original)
    aug_results.append((f"Brightness ({val})", cv2.cvtColor(brt_img, cv2.COLOR_BGR2RGB)))
    
    # (4) Noise
    noise_img = add_noise(original)
    aug_results.append(("Gaussian Noise", cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)))
    
    # (5) Blur
    blur_img, ksize = add_blur(original)
    aug_results.append((f"Motion Blur (k={ksize})", cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)))

    # 4. 시각화 (2행 3열)
    plt.figure(figsize=(12, 8))
    
    # 원본
    plt.subplot(2, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Original")
    plt.axis('off')
    
    # 증강 결과들
    for i, (title, img) in enumerate(aug_results):
        plt.subplot(2, 3, i+2)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
