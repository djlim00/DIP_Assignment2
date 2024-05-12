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

def gaussian_kernel(size, sigma=1.0):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def gaussian_filter(img, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)
    gauss_img = convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=0)
    return gauss_img

# 이미지 로드 및 그레이스케일 변환
#img = np.array(Image.open('resource\\HW3 Image Samples\\HW3 Image Samples\\Noise Filtering\\Fig0503 (original_pattern).jpg').convert('L'))
#img = np.array(Image.open('resource\\HW3 Image Samples\\HW3 Image Samples\\Noise Filtering\\Fig0504(a)(gaussian-noise).jpg').convert('L'))
img = np.array(Image.open('resource\\HW3 Image Samples\\HW3 Image Samples\\Noise Filtering\\Fig0504(i)(salt-pepper-noise).jpg').convert('L'))
#img = np.array(Image.open('resource\\HW3 Image Samples\\HW3 Image Samples\\Noise Filtering\\Gaussian noise.png').convert('L'))
#img = np.array(Image.open('resource\\HW3 Image Samples\\HW3 Image Samples\\Noise Filtering\\Lena_noise.png').convert('L'))
#img = np.array(Image.open('resource\\HW3 Image Samples\\HW3 Image Samples\\Noise Filtering\\Salt&pepper noise.png').convert('L'))



# Median Filter 적용
median_img = median_filter(img, 3)  # Kernel size = 3

# Gaussian Filter 적용
gauss_img = gaussian_filter(img, 5, 1)  # Kernel size = 5, Sigma = 1

# 결과 출력
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Median Filter")
plt.imshow(median_img, cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Gaussian Filter")
plt.imshow(gauss_img, cmap='gray')
plt.show()
