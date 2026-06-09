# Assignment 2 - Computer Vision
# Name: Gustavo Nunes Lopes

import os
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv
import os

########################################################################################################################
# Function to normalize points
# Input: points (image points to be normalized)
# Output: norm_points (normalized points)
#         T (normalization matrix)

def normalize_points(points):
    points = to_homogeneous(points)
    points = points / points[:, 2].reshape(-1, 1)

    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])

    avg_distance = np.mean(np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2))

    if avg_distance == 0:
        raise ValueError("Cannot normalize coincident points")

    scale = np.sqrt(2) / avg_distance

    T = np.array([[ scale,   0,    -scale*cx ],
                  [ 0,     scale,  -scale*cy ],
                  [ 0,       0,        1     ]])

    norm_pts = (T @ points.T).T

    return norm_pts, T

# Function to build the matrix A of the DLT equation system
# Input: pts1, pts2 (points "pts1" from the first image and points "pts2" from the second image satisfying pts2 = H.pts1)
# Output: A (matrix containing the two or three rows resulting from the relation pts2 × H.pts1 = 0)

def compute_A(pts1, pts2):
    n = pts1.shape[0]
    A = np.zeros((2 * n, 9))

    for i in range(n):
        x,  y,  w  = pts1[i]
        xp, yp, wp = pts2[i]

        A[2*i]   = [   0,    0,    0,  -wp*x, -wp*y, -wp*w,  yp*x,  yp*y,  yp*w]
        A[2*i+1] = [ wp*x,  wp*y,  wp*w,   0,     0,     0, -xp*x, -xp*y, -xp*w]
    return A

# Normalized DLT function
# Input: pts1, pts2 (points "pts1" from the first image and points "pts2" from the second image satisfying pts2 = H.pts1)
# Output: H (estimated homography matrix)

def compute_normalized_dlt(pts1, pts2):
    norm_pts1, T1 = normalize_points(pts1)
    norm_pts2, T2 = normalize_points(pts2)

    A = compute_A(norm_pts1, norm_pts2)

    U, S, Vt = np.linalg.svd(A)

    h = Vt[-1]
    H_normalized = h.reshape(3, 3)

    H = np.linalg.inv(T2) @ H_normalized @ T1

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    return H

def find_inliers(pts1, pts2, H, dis_threshold):
    pts1 = to_homogeneous(pts1)
    pts2 = to_homogeneous(pts2)

    pts2 = pts2 / pts2[:, 2].reshape(-1, 1)

    projected_pts2 = (H @ pts1.T).T
    valid = np.abs(projected_pts2[:, 2]) > 1e-12

    distances = np.full(pts1.shape[0], np.inf)

    projected_pts2[valid] = projected_pts2[valid] / projected_pts2[valid, 2].reshape(-1, 1)
    distances[valid] = np.sqrt(
        (projected_pts2[valid, 0] - pts2[valid, 0]) ** 2 +
        (projected_pts2[valid, 1] - pts2[valid, 1]) ** 2
    )

    inliers = distances < dis_threshold

    return inliers

def to_homogeneous(points):
    points = np.asarray(points, dtype=np.float64)

    if points.ndim == 3:
        points = points.reshape(-1, points.shape[-1])

    if points.shape[1] == 2:
        ones = np.ones((points.shape[0], 1))
        points = np.hstack([points, ones])

    return points

# RANSAC function
# Inputs:
# pts1: points from the first image
# pts2: points from the second image
# dis_threshold: distance threshold to be used in RANSAC
# N: maximum number of iterations (it may be defined inside the function and should be updated
#    dynamically according to the number of inliers/outliers)
# Ninl: desired inlier threshold (it may or may not be used — this is your choice)
# Outputs:
# H: estimated homography
# pts1_in, pts2_in: set of inlier points from the first and second images

def RANSAC(pts1, pts2, dis_threshold, N, Ninl):
    pts1 = to_homogeneous(pts1)
    pts2 = to_homogeneous(pts2)

    num_pts = pts1.shape[0]

    max_inliers = 0
    best_H = None
    best_pts1_in = None
    best_pts2_in = None

    i = 0

    while i < N:
        indices = np.random.choice(num_pts, 4, replace=False)

        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]

        try:
            H = compute_normalized_dlt(sample_pts1, sample_pts2)
        except (np.linalg.LinAlgError, ValueError):
            i += 1
            continue

        if not np.all(np.isfinite(H)):
            i += 1
            continue

        inliers = find_inliers(pts1, pts2, H, dis_threshold)
        num_inliers = np.sum(inliers)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H
            best_pts1_in = pts1[inliers]
            best_pts2_in = pts2[inliers]

        i += 1

    if best_pts1_in is not None and len(best_pts1_in) >= 4:
        best_H = compute_normalized_dlt(best_pts1_in, best_pts2_in)

    print('Inliers:', max_inliers)
    print('Iterações:', i)

    return best_H, best_pts1_in, best_pts2_in

########################################################################################################################
# Application example for testing your function for estimating homography

PATH_IMAGES = os.path.join(os.path.dirname(__file__), '..', 'images')
PATH_QUERY_IMAGE = os.path.join(PATH_IMAGES, 'box.jpg')
PATH_TRAIN_IMAGE = os.path.join(PATH_IMAGES, 'photo01a.jpg')

MIN_MATCH_COUNT = 10
img1 = cv.imread(PATH_QUERY_IMAGE, 0)   # queryImage
img2 = cv.imread(PATH_TRAIN_IMAGE, 0)        # trainImage

# SIFT initialization
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


# FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])
    
    #################################################
    M, pts1_in, pts2_in = RANSAC(src_pts, dst_pts, dis_threshold=5, N=1000, Ninl=10)
    #################################################

    img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0])) 

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

fig, axs = plt.subplots(2, 2, figsize=(30, 15))
fig.add_subplot(2, 2, 1)
plt.imshow(img3, 'gray')
fig.add_subplot(2, 2, 2)
plt.title('First image')
plt.imshow(img1, 'gray')
fig.add_subplot(2, 2, 3)
plt.title('Second image')
plt.imshow(img2, 'gray')
fig.add_subplot(2, 2, 4)
plt.title('First image after warping')
plt.imshow(img4, 'gray')
plt.show()

########################################################################################################################