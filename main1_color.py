import cv2
import matplotlib.pyplot as plt

# 이미지 로드
img = cv2.imread('resource\\HW3 Image Samples\\HW3 Image Samples\\Noise Filtering\\Lena_noise.png')  # 이미지 경로를 올바르게 설정해주세요.

# OpenCV는 BGR 순서로 이미지를 로드하므로 RGB로 변환
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 메디안 필터 적용
median_img = cv2.medianBlur(img_rgb, 7)

# 가우시안 필터 적용
gaussian_img = cv2.GaussianBlur(img_rgb, (7, 7), 10)

# 결과 출력
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(median_img)
plt.title("Median Filter")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gaussian_img)
plt.title("Gaussian Filter")
plt.axis('off')

plt.show()
