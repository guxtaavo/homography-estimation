# Assignment 2 - Computer Vision
# Name: Gustavo Nunes Lopes

# Import the required libraries
# Add any other libraries you want

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv


########################################################################################################################
# Function to normalize points
# Input: points (image points to be normalized)
# Output: norm_points (normalized points)
#         T (normalization matrix)

def normalize_points(points):
    
    
    
    return norm_points, T

# Function to build the matrix A of the DLT equation system
# Input: pts1, pts2 (points "pts1" from the first image and points "pts2" from the second image satisfying pts2 = H.pts1)
# Output: A (matrix containing the two or three rows resulting from the relation pts2 × H.pts1 = 0)

def compute_A(pts1, pts2):
    
    return A


# Normalized DLT function
# Input: pts1, pts2 (points "pts1" from the first image and points "pts2" from the second image satisfying pts2 = H.pts1)
# Output: H (estimated homography matrix)

def compute_normalized_dlt(pts1, pts2):

    # Normalize points
    
    # Build the equation system by stacking the matrix A for each pair of normalized corresponding points
    
    # Compute the SVD of the stacked matrix A and estimate the normalized homography H_normalized
    
    # Denormalize H_normalized to obtain H
    


    return H

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
    
    # Define other parameters such as the number of model samples, probabilities used in the equation for N, etc.
    

    # Loop
    
        # While the stopping criterion is not satisfied 
        
        
        # Randomly select "s" samples from the set of point pairs pts1 and pts2 
        
        
        # Use the samples to estimate a homography using the Normalized DLT 
        
        
        # Test this homography with the remaining point pairs using dis_threshold and count 
        # the number of supposed inliers obtained with the estimated model
        
        # If the number of inliers is the largest obtained so far, store this set along with 
        # the "s" samples used.       
        # Also update the required number N of iterations
        

    # After exiting the loop
    # Estimate the final homography H using all selected inliers.
    

    return H, pts1_in, pts2_in


########################################################################################################################
# Application example for testing your function for estimating homography


MIN_MATCH_COUNT = 10
img1 = cv.imread('box.jpg', 0)   # queryImage
img2 = cv.imread('photo01a.jpg', 0)        # trainImage

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
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
    
    #################################################
    M = # your function !!!!
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