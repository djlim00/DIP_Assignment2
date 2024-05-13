from skimage import io, color, filters, feature
import numpy as np
import matplotlib.pyplot as plt


#img_path = 'resource\\HW3 Image Samples\\HW3 Image Samples\\Edge Detection\\Fig0327(a)(tungsten_original).jpg'
img_path = 'resource\\HW3 Image Samples\\HW3 Image Samples\\Edge Detection\\lenna_color.bmp'

img = io.imread(img_path)

if len(img.shape) == 3 and img.shape[2] == 3:
    img_gray = color.rgb2gray(img)
else:
    img_gray = img

# Sobel 
edges_sobel = filters.sobel(img_gray)

# Prewitt 
edges_prewitt = filters.prewitt(img_gray)

# Canny 
edges_canny = feature.canny(img_gray)

# LoG
img_gaussian = filters.gaussian(img_gray, sigma=2)
edges_log = filters.laplace(img_gaussian)

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