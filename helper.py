import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


# helper function to undistort and transofrm
def plot_points(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img, cmap='gray')
    # plt.title('points chosen manually')
    plt.plot(1180, 500, 'o')  # topright
    plt.plot(1180, 700, 'o')  # bottomright
    plt.plot(270, 500, 'o')  # topleft
    plt.plot(270, 700, 'o')  # bottomleft
    plt.show()


def color_thresh(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow_min = np.array([0, 80, 200], np.uint8)
    yellow_max = np.array([40, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(img, yellow_min, yellow_max)

    white_min = np.array([20, 0, 200], np.uint8)
    white_max = np.array([255, 80, 255], np.uint8)
    white_mask = cv2.inRange(img, white_min, white_max)

    binary_output = np.zeros_like(img[:, :, 0])
    binary_output[((yellow_mask != 0) | (white_mask != 0))] = 1

    filtered = img
    filtered[((yellow_mask == 0) & (white_mask == 0))] = 0

    return binary_output


def subplot_images(img1, img2, title1, title2):

    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    f.tight_layout(pad=3)

    ax1.imshow(img1, cmap='Greys')
    ax1.set_title(title1, fontsize=50)
    ax2.imshow(img2, cmap='Greys')
    ax2.set_title(title2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def cal_undistort(img, mtx, dist):

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst


# warp image to get a bird eye view
def warp(img, src, dst):
    # grayscale image
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    minV = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M, minV


# unwarp image to original perspective
def unwarp(warp_img, minV):
    img_size = (warp_img.shape[1], warp_img.shape[0])
    unwarped = cv2.warpPerspective(
        warp_img, minV, img_size, flags=cv2.INTER_LINEAR)
    return unwarped


# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, ksize=3, orient='x', thresh_min=130, thresh_max=255):
    # # Apply the following steps to img
    # # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if (orient == 'x'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    if (orient == 'y'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_label = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # 6) Return this mask as your binary_output image

    binary_output = np.zeros_like(scaled_label)
    binary_output[(scaled_label >= thresh_min) &
                  (scaled_label <= thresh_max)] = 1
    return binary_output


# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # # Apply the following steps to img
    # # 1) Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled = (np.max(magnitude) / 255).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    mask = np.zeros_like(scaled)
    mask[(scaled > mag_thresh[0]) & (scaled < mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    binary_output = mask
    return binary_output


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    magnitx = np.absolute(sobelx)
    magnity = np.absolute(sobely)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of
    # the gradient
    arctan = np.arctan2(magnity, magnitx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(arctan)
    binary_output[(arctan >= thresh[0]) & (arctan <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(90, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 2) Apply a threshold to the S channel
    s = hls[:, :, 2]
    # 3) Return a binary image of threshold result
    binary = np.zeros_like(s)
    binary[(s > thresh[0]) & (s <= thresh[1])] = 1  # placeholder line
    return binary


# helper for thresh
def thresh(img, thresh_min, thresh_max):
    ret = np.zeros_like(img)
    ret[(img >= thresh_min) & (img <= thresh_max)] = 1
    return ret


def hsv_select(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    t_yellow_H = thresh(H, 10, 30)
    t_yellow_S = thresh(S, 50, 255)
    t_yellow_V = thresh(V, 150, 255)

    t_white_R = thresh(R, 225, 255)
    t_white_V = thresh(V, 230, 255)

    b = np.zeros((img.shape[0], img.shape[1]))
    b[(t_yellow_H == 1) & (t_yellow_S == 1) & (t_yellow_V == 1)] = 1
    b[(t_white_R == 1) | (t_white_V == 1)] = 1
    return b
