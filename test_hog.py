import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import os

# --- 테스트할 이미지 경로 설정 ---
# 라벨링한 이미지 중 하나를 선택해서 경로를 넣어주세요.
# 예: "./dataset/positive/pos_00000.jpg" 또는 새로 찍은 사진
TEST_IMAGE_PATH = "./positive/pos_00001.jpg" 

# HOG 파라미터 (extract_features.py와 동일하게 설정하는 것이 좋습니다)
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (12, 12),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'transform_sqrt': True
    # visualize=True 옵션은 아래 함수 호출 시 따로 넣습니다.
}

def visualize_hog(image_path):
    # 1. 이미지 읽기
    if not os.path.exists(image_path):
        print(f"Error: 파일을 찾을 수 없습니다: {image_path}")
        print("TEST_IMAGE_PATH 변수에 올바른 이미지 경로를 입력해주세요.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Error: 이미지를 열 수 없습니다.")
        return

    # 2. 흑백 변환 및 리사이즈 (필수 전처리)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(gray, (128, 128))

    # 3. HOG 특징 및 시각화 이미지 추출
    # visualize=True를 주면 특징 벡터(fd)와 함께 시각화 이미지(hog_image)를 반환합니다.
    fd, hog_image = hog(resized_gray, **HOG_PARAMS, visualize=True)

    print(f"원본 이미지 크기: {img.shape}")
    print(f"리사이즈된 이미지 크기: {resized_gray.shape}")
    print(f"추출된 HOG 특징 벡터 길이: {len(fd)}")

    # --- 시각화 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    # 원본 (리사이즈된 흑백) 이미지 표시
    ax1.axis('off')
    ax1.imshow(resized_gray, cmap=plt.cm.gray)
    ax1.set_title('Resized Gray Image (128x128)')

    # HOG 이미지 표시
    # HOG 이미지는 픽셀 값이 매우 작아서 잘 보이도록 조정해줍니다.
    from skimage import exposure
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('HOG Visualization')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 경로에 본인이 가지고 있는 이미지 파일 경로를 넣고 실행해보세요.
    # 예: 라벨링해둔 파일이 있다면 그 경로를 입력하세요.
    # 만약 파일이 없다면 에러 메시지가 출력됩니다.
    
    # 예시 경로 (본인의 경로로 수정 필요!)
    example_path = TEST_IMAGE_PATH
    
    # 경로가 실제로 존재할 때만 실행
    if os.path.exists(example_path):
        visualize_hog(example_path)
    else:
        print(f"알림: '{example_path}' 경로에 파일이 없습니다.")
        print("TEST_IMAGE_PATH 변수에 테스트하고 싶은 이미지의 정확한 경로를 입력해주세요.")
        # 임시로 빈 캔버스를 만들어 코드가 돌아가는지 확인할 수도 있습니다.
        # dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)
        # cv2.imwrite("dummy_test.jpg", dummy_img)
        # visualize_hog("dummy_test.jpg")