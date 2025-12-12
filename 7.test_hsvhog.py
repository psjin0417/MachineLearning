import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import os

TEST_IMAGE_PATH = "/home/autonav/MachineLearning/positive/pos_00002.jpg" 

# HOG 파라미터
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (12, 12),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'transform_sqrt': True,
    # channel_axis: 입력 이미지가 컬러(3채널)일 때 마지막 축을 채널로 인식
    # None이면 흑백(2차원)으로 처리
    'channel_axis': None 
}

def visualize_hog_hsv(image_path, mode='gray'):
    if not os.path.exists(image_path):
        print("파일 없음")
        return

    img = cv2.imread(image_path)
    if img is None: return

    # --- 전처리 분기 ---
    if mode == 'hsv_v':
        # HSV로 변환 후 V채널만 사용 (흑백과 비슷하지만 조명에 더 강할 수 있음)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        target_img = cv2.resize(hsv[:, :, 2], (128, 128)) # Index 2 = V channel
        HOG_PARAMS['channel_axis'] = None # 단일 채널이므로 None
        title_text = "HSV - V Channel"
        
    elif mode == 'hsv_all':
        # H, S, V 모든 채널을 다 사용하여 가장 강한 특징 추출 (Feature Length는 동일)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        target_img = cv2.resize(hsv, (128, 128))
        HOG_PARAMS['channel_axis'] = -1 # 3채널(Multi-channel) 사용 설정
        title_text = "HSV - All Channels"
        
    else: # default 'gray'
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        target_img = cv2.resize(gray, (128, 128))
        HOG_PARAMS['channel_axis'] = None
        title_text = "Grayscale"

    # HOG 추출
    fd, hog_image = hog(target_img, **HOG_PARAMS, visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # --- 시각화 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 왼쪽: 입력 이미지 보여주기
    ax1.axis('off')
    if mode == 'hsv_all':
        # matplotlib는 HSV 출력을 지원 안하므로 RGB로 바꿔서 보여줌 (확인용)
        ax1.imshow(cv2.cvtColor(target_img, cv2.COLOR_HSV2RGB))
    else:
        ax1.imshow(target_img, cmap='gray')
    ax1.set_title(f'Input: {title_text} (128x128)', fontsize=16)

    # 오른쪽: HOG 이미지
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title(f'HOG Visualization ({mode})', fontsize=16)

    # 정보 텍스트
    info_text = (f"■ HOG INFO ({mode})\n"
                 f"----------------------\n"
                 f" Feature Length : {len(fd)}\n"
                 f" Input Shape    : {target_img.shape}\n"
                 f" Channel Mode   : {'Multi (All)' if HOG_PARAMS['channel_axis'] else 'Single'}")

    ax2.text(0.03, 0.97, info_text, transform=ax2.transAxes, fontsize=15, 
             fontweight='bold', verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.85))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 원하는 모드를 선택해서 실행해보세요
    # 1. 'gray'    : 일반적인 흑백
    # 2. 'hsv_v'   : HSV중 V만 사용 (추천)
    # 3. 'hsv_all' : 3채널 모두 고려 (느리지만 정보량 많음)
    
    visualize_hog_hsv(TEST_IMAGE_PATH, mode='hsv_v')