import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image

def gaussian_kernel(size, sigma=1.0):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def gaussian_blur(img, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_img = convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=0)
    return blurred_img

def high_boost_filter(img, kernel_size, sigma, alpha):
    blurred_img = gaussian_blur(img, kernel_size, sigma)
    mask = img - blurred_img
    high_boosted_img = img + alpha * mask
    return high_boosted_img

# 이미지 로드 및 그레이스케일 변환
#img = np.array(Image.open('resource\\HW3 Image Samples\\HW3 Image Samples\\High-boost Filtering\\Fig0327(a)(tungsten_original).jpg').convert('L'))
img = np.array(Image.open('resource\\HW3 Image Samples\\HW3 Image Samples\\High-boost Filtering\\Fig0525(a)(aerial_view_no_turb).jpg').convert('L'))

# 다양한 alpha 값으로 High-boost 필터링 적용
alphas = [0.5, 1, 1.5, 2, 2.5]

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

for i, alpha in enumerate(alphas, start=2):
    high_boosted_img = high_boost_filter(img, 5, 1, alpha)
    
    plt.subplot(2, 3, i)
    plt.title(f"High-boost Filter: Alpha={alpha}")
    plt.imshow(high_boosted_img, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()
