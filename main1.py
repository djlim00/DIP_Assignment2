import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image

def median_filter(img, kernel_size):
    pad_size = kernel_size // 2
    padded_img = np.pad(img, pad_size, mode='constant', constant_values=0)
    median_img = np.zeros_like(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            median_img[i, j] = np.median(padded_img[i:i+kernel_size, j:j+kernel_size])
    return median_img

def gaussian_kernel(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def gaussian_filter(img, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)
    gauss_img = convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=0)
    return gauss_img

def median_filter_color(img, kernel_size):
    # 채널별로 median 필터 적용
    median_img = np.zeros_like(img)
    for c in range(3):  # RGB 채널
        median_img[:, :, c] = median_filter(img[:, :, c], kernel_size)
    return median_img

def gaussian_filter_color(img, kernel_size, sigma):
    # 채널별로 Gaussian 필터 적용
    gauss_img = np.zeros_like(img)
    for c in range(3):  # RGB 채널
        gauss_img[:, :, c] = gaussian_filter(img[:, :, c], kernel_size, sigma)
    return gauss_img

# 이미지 로드 부분 수정 (convert('L') 제거하여 컬러 이미지로 로드)
img = np.array(Image.open('resource\HW3 Image Samples\HW3 Image Samples\Noise Filtering\Fig0503 (original_pattern).jpg').convert('L'))
#img = np.array(Image.open('resource\HW3 Image Samples\HW3 Image Samples\Noise Filtering\Fig0504(a)(gaussian-noise).jpg').convert('L'))
#img = np.array(Image.open('resource\HW3 Image Samples\HW3 Image Samples\Noise Filtering\Fig0504(i)(salt-pepper-noise).jpg').convert('L'))

# 컬러 이미지인 경우 필터링
if len(img.shape) == 3:  # 컬러 이미지 확인
    median_img = median_filter_color(img, 5)  # Kernel size = 5
    gauss_img = gaussian_filter_color(img, 5, 10)  # Kernel size = 5, Sigma = 10
else:  # 흑백 이미지인 경우 (기존 코드 사용)
    median_img = median_filter(img, 5)
    gauss_img = gaussian_filter(img, 5, 10)

# 결과 출력 부분의 cmap 옵션 제거하여 컬러 이미지로 출력
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img)  # 컬러 이미지로 출력
plt.subplot(1, 3, 2)
plt.title("Median Filter")
plt.imshow(median_img)  # 컬러 이미지로 출력
plt.subplot(1, 3, 3)
plt.title("Gaussian Filter")
plt.imshow(gauss_img)  # 컬러 이미지로 출력
plt.show()
