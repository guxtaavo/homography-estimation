from src.assingment02_template import RANSAC
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

def main():
    # Application example for testing your function for estimating homography
    PATH_IMAGES = os.path.join(os.path.dirname(__file__), 'images')
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

if __name__ == "__main__":
    main()
