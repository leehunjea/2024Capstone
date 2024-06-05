from __future__ import print_function
import cv2 as cv

from matplotlib import pyplot as plt

img1 = cv.imread('Test4.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('Test5.jpg', cv.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)

# Initiate SIFT detector
detector = cv.xfeatures2d_SIFT.create()

keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

# FLANN paramters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(check=50) # or pass dictory

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))] # python2.x for xrange()

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.5 * n.distance:
        matchesMask[i] = [1,0]

draw_params = dict(matchColor=(0, 0, 255), singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask, flags=0)

img_matches = cv.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches, None, **draw_params)

#-- Show detected matches
cv.imshow('Matches', img_matches)
cv.waitKey(0)
