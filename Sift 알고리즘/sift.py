from __future__ import print_function
import cv2 as cv
import numpy as np

img1 = cv.imread('Test.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('Test5.jpg', cv.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)

#-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
detector = cv.xfeatures2d_SIFT.create()

keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

#-- Step 2: Matching descriptor vectors with a brute force matcher
matcher = cv.BFMatcher()
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

#--Step 3: Apply Ratio Test
good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append([m])

#-- Draw matches
img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
cv.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good, img_matches)

#-- Show detected matches
cv.imshow('Matches', img_matches)
cv.waitKey(0)
