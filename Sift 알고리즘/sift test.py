import cv2
import numpy as np

# SIFT 특징 추출기 생성
sift = cv2.SIFT_create()

# 두 이미지 로드
img1 = cv2.imread('Test5.jpg')
img2 = cv2.imread('Test4.jpg')

# 특징점 추출
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 특징점 매칭
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 좋은 매칭 포인트만 필터링
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 변화 영역 마스크 생성
mask = np.zeros_like(img1)
for match in good_matches:
    pt1 = kp1[match.queryIdx].pt
    pt2 = kp2[match.trainIdx].pt
    mask = cv2.line(mask, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)

# 변화 영역 시각화
img_diff = cv2.absdiff(img1, img2)
img_diff[mask == 0] = 0  # 변화 없는 영역은 검정색으로 마스킹
cv2.imshow('Change Detection', img_diff)
cv2.waitKey(0)
cv2.destroyAllWindows()