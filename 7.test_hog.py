import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import os

# --- 테스트할 이미지 경로 설정 ---
TEST_IMAGE_PATH = "/home/autonav/MachineLearning/positive/pos_00002.jpg" 

# HOG 파라미터
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (12, 12),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'transform_sqrt': True
}

def visualize_hog(image_path):
    if not os.path.exists(image_path):
        print(f"Error: 파일을 찾을 수 없습니다: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Error: 이미지를 열 수 없습니다.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(gray, (128, 128))

    fd, hog_image = hog(resized_gray, **HOG_PARAMS, visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # --- [변경 1] 전체 창 크기(figsize)를 키움 (12, 6 -> 16, 8) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    # 원본 이미지
    ax1.axis('off')
    ax1.imshow(resized_gray, cmap=plt.cm.gray)
    ax1.set_title(f'Original (Resized 128x128)', fontsize=16) # 제목 폰트도 키움

    # HOG 이미지
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('HOG Visualization', fontsize=16) # 제목 폰트도 키움

    # --- [변경 2] 텍스트 정보를 더 크고 진하게 표시 ---
    info_text = (f"■ HOG PARAMETERS\n"
                 f"----------------------\n"
                 f" Orientations   : {HOG_PARAMS['orientations']}\n"
                 f" Pixels/Cell    : {HOG_PARAMS['pixels_per_cell']}\n"
                 f" Cells/Block    : {HOG_PARAMS['cells_per_block']}\n"
                 f" Transform Sqrt : {HOG_PARAMS['transform_sqrt']}")

    # fontsize를 10에서 15로 키우고, 박스 패딩(pad)을 넉넉하게 주었습니다.
    ax2.text(0.03, 0.97, info_text, 
             transform=ax2.transAxes, 
             fontsize=15,               # 글자 크기 확대
             fontweight='bold',         # 글자 굵게
             verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.85, edgecolor='black'))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if os.path.exists(TEST_IMAGE_PATH):
        visualize_hog(TEST_IMAGE_PATH)
    else:
        print(f"경로를 확인해주세요: {TEST_IMAGE_PATH}")