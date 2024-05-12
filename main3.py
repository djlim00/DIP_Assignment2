from skimage import io, color, filters, feature
import numpy as np
import matplotlib.pyplot as plt

# 이미지를 읽어옵니다.
#img_path = 'resource\\HW3 Image Samples\\HW3 Image Samples\\Edge Detection\\Fig0327(a)(tungsten_original).jpg'
img_path = 'resource\\HW3 Image Samples\\HW3 Image Samples\\Edge Detection\\lenna_color.bmp'

img = io.imread(img_path)

# 이미지가 그레이스케일인지 확인합니다. RGB 이미지는 3차원이고 그레이스케일 이미지는 2차원입니다.
if len(img.shape) == 3 and img.shape[2] == 3:
    # 이미지가 RGB라면 그레이스케일로 변환합니다.
    img_gray = color.rgb2gray(img)
else:
    # 이미지가 이미 그레이스케일이라면 그대로 사용합니다.
    img_gray = img

# Sobel 에지 검출
edges_sobel = filters.sobel(img_gray)

# Prewitt 에지 검출
edges_prewitt = filters.prewitt(img_gray)

# Canny 에지 검출
edges_canny = feature.canny(img_gray)

# LoG(Laplacian of Gaussian) 에지 검출
# LoG를 직접 구현하는 대신, 먼저 Gaussian smoothing을 적용한 후 Laplace 함수를 사용합니다.
img_gaussian = filters.gaussian(img_gray, sigma=2)
edges_log = filters.laplace(img_gaussian)

# 결과 시각화
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img_gray, cmap=plt.cm.gray)
ax[0].set_title('Original Image')

ax[1].imshow(edges_sobel, cmap=plt.cm.gray)
ax[1].set_title('Sobel Edge Detection')

ax[2].imshow(edges_prewitt, cmap=plt.cm.gray)
ax[2].set_title('Prewitt Edge Detection')

ax[3].imshow(edges_canny, cmap=plt.cm.gray)
ax[3].set_title('Canny Edge Detection')

ax[4].imshow(edges_log, cmap=plt.cm.gray)
ax[4].set_title('LoG Edge Detection')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
