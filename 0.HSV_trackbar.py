import cv2
import numpy as np

def on_trackbar_change(val):
    pass  # 이 함수는 트랙바 값이 바뀔 때마다 호출됩니다. 여기서는 아무 작업도 수행하지 않습니다.

def main():
    # 이미지 로드
    image = cv2.imread('/home/autonav/MachineLearning/positive/pos_00002.jpg')

    # 윈도우 생성
    cv2.namedWindow("Image")

    # 트랙바 생성
    cv2.createTrackbar("H_low", "Image", 0, 179, on_trackbar_change)
    cv2.createTrackbar("H_high", "Image", 179, 179, on_trackbar_change)
    cv2.createTrackbar("S_low", "Image", 0, 255, on_trackbar_change)
    cv2.createTrackbar("S_high", "Image", 255, 255, on_trackbar_change)
    cv2.createTrackbar("V_low", "Image", 0, 255, on_trackbar_change)
    cv2.createTrackbar("V_high", "Image", 255, 255, on_trackbar_change)

    while True:
        # 트랙바 값을 읽어서 HSV 범위 설정
        h_low = cv2.getTrackbarPos("H_low", "Image")
        h_high = cv2.getTrackbarPos("H_high", "Image")
        s_low = cv2.getTrackbarPos("S_low", "Image")
        s_high = cv2.getTrackbarPos("S_high", "Image")
        v_low = cv2.getTrackbarPos("V_low", "Image")
        v_high = cv2.getTrackbarPos("V_high", "Image")

        # HSV 범위로 마스크 생성
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([h_low, s_low, v_low])
        upper_bound = np.array([h_high, s_high, v_high])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # 마스크를 이용하여 원본 이미지에 색상 표시
        result = cv2.bitwise_and(image, image, mask=mask)

        # 결과 이미지 출력
        cv2.imshow("Image", result)

        # 키 입력 대기
        key = cv2.waitKey(1)
        if key == 27:  # ESC 키를 누르면 종료
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()