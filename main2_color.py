import cv2
import numpy as np

def high_boost_filtering(image_path, alpha=1.5):
    # 이미지를 컬러로 로드
    image = cv2.imread(image_path)
    
    # 이미지를 블러 처리 (낮은 주파수 세부 사항 제거)
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    
    # 마스크 생성: 원본 이미지에서 블러 이미지를 빼서 고주파 세부 사항 추출
    mask = cv2.subtract(image, blurred)
    
    # High-boost: 원본 이미지에 고주파 세부 사항을 (1+alpha)만큼 강화하여 추가
    high_boosted = cv2.addWeighted(image, alpha, mask, 1, 0)
    
    return high_boosted

# 이미지 경로
image_path = 'path/to/your/image.jpg'

# High-boost 필터링 적용
high_boosted_image = high_boost_filtering(image_path, alpha=1.5)

# 결과 이미지 표시
cv2.imshow('Original Image', cv2.imread(image_path))
cv2.imshow('High-Boost Filtered Image', high_boosted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()