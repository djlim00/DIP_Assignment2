import cv2
import numpy as np

def high_boost_filtering(image, alpha=1.5):
    
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    
    mask = cv2.subtract(image, blurred)
    
    # High-boost: 원본 이미지에 고주파 세부 사항을 (1+alpha)만큼 강화하여 추가
    high_boosted = cv2.addWeighted(image, alpha, mask, 1, 0)
    
    return high_boosted

image_path = 'resource\\HW3 Image Samples\\HW3 Image Samples\\Edge Detection\\lenna_color.bmp'


image = cv2.imread(image_path)


high_boosted_image = high_boost_filtering(image, alpha=1.0)


combined_image = np.hstack((image, high_boosted_image))


cv2.imshow('Original Image and High-Boost Filtered Image', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
