from helper import *
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import glob



#import pickle file
calib_dist = pickle.load(open('./camera_cal/calib_dist_mtx.p',"rb"))
mtx = calib_dist["mtx"]
dist = calib_dist["dist"]

#import test images
images = glob.glob('./test_images/*.jpg')

#loop over each image and apply preprocessing pipeline
for idx, fname in enumerate(images):

    img = cv2.imread(fname)  #read image
    print('workin on ', fname)

    undis_img =cal_undistort(img, mtx, dist)# undistort each image

    #define points for perspective transformation
    src = np.float32([[270, 700], [540, 500], [1120, 700], [800, 500]])
    dst = np.float32([[270, 700], [270, 500], [1120, 700], [1120, 500]])
    # color threshold
    s_channel = hls_select(undis_img, thresh=(120, 255)) #this threshold is shown with a cleanest line extraction
    #warp
    binary, M, minV = warp(s_channel, src, dst)

    #now clean lines are extracted lets go find the lanes
    histogram = np.sum(binary[int(binary.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary, binary, binary)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary.shape[0] / nwindows)
    #non zero pixels
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary.shape[0] - (window+1)*window_height
        win_y_high = binary.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary.shape[0] - 1, binary.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    #calculate radius of carvature
    y_eval = np.max(ploty) #maximum y-value, corresponding to the bottom of the image
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    print(left_curverad, right_curverad)
    # # Define conversions in x and y from pixels space to meters
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm') # test values make sense 387.596834937 m 395.13117416 m
    #ok lets unwarp and draw line on image
    warp_zero = np.zeros_like(binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    #unwarp
    # Combine the result with the original image
    left_lane = np.array(list(zip(np.concatenate((left_fitx - margin / 2, left_fitx[::-1] + margin / 2), axis=0),
                                  np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx - margin / 2, right_fitx[::-1] + margin / 2), axis=0),
                                   np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)
    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)
    cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road_bkg, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(road_bkg, [right_lane], color=[0, 0, 255])
    road = unwarp(road, minV)
    road_bk = unwarp(road_bkg, minV)
    base = cv2.addWeighted(img, 1.0, road_bk, -1.0, 0.0)
    base = cv2.addWeighted(img, 1.0, road, 2.0, 0.0)

    #write radius and vehicle position difference from center of the lane
    camera_center = (left_fitx[-1]+right_fitx[-1])/2
    center_diff = (camera_center- binary.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos='right'

    cv2.putText(base,'Radius of Curvature is: '+str(round(left_curverad,3)) +'(m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(base,'Vehicle is: '+str(abs(round(center_diff,3))) +'m '+side_pos +' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    write_to_fie = './imgs_tracked/test_tracked'+str(idx)+'.jpg'
    cv2.imwrite(write_to_fie,base)


