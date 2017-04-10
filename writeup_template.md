##Writeup Template


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The project consist of following files:

* `camera_calib.py`: calibrating images using 6 * 9 corner chess boards and saving it to a pickle file
* `camera_calib/calib_dist_mtx.p`: saved pickle file for distortion coefficients and calibration matrix
* `pipeline.py`: pipeline to process images will draw detected line on original image
* `helper.py` : helper functions to use during pipeline 
* `video_gen.py` : generate a video result for detected lane lines in a video

[//]: # (Image References)

[image1]: ./output_images/chess_undist.png "Undistorted"
[image2]: ./output_images/undistorted/undistorted5.jpg "Road Transformed"
[image3]: ./output_images/threshold.png "Binary Example"
[image4]: ./output_images/points.png "points Example"
[image5]: ./output_images/warp-b.png "warp Example"
[image6]: ./output_images/hist.png "histogram Example"
[image7]: ./output_images/windows.png "window Example"
[image8]: ./imgs_tracked/test_tracked0.jpg "result"



[video1]: ./tracked_video.mp4 "Video"

---
###Writeup / README

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `camera_calib.py` 

I start by preparing "object points", which will be the standard points in the world, we will fill a 2d array for `x=6` and `y=9` so that means we populate points from (0,0,0) .....(8,5,0) in `objp` object. For finding image points Opencv function `cv2.findChessboardCorners()` is used. this will return corners/image points. I then append each `objp` and `corners` to the `img_points` and `obj_points` list. 

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

in the end I dumped calculated values to a pickle file.

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
you can find the function `cal_undistort(img, mtx, dist)` in `helper.py` 
 
First I load the pickle file to extract matrix corners and distortion coefficients and then I use `cv2.undistort` function to undistort images, for `mtx` and `dist` I should pass the data from pickle file loaded earlier because that has the right values for calibration.
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of saturate channel color(S in HLS) and  gradient in X direction (using `cv2.Sobel` function) to generate a binary image (thresholding steps at 35 to 37 in `pipeline.py`). Sobel X operator with kernel size `ksize = 15` has shown robust result to extract parallel lines cleanly with thresholding ratio between `min = 15` and `max = 100`. Moreover I combine X gradient image withthresholded S channel. For S channel I used threshold between `min=90` and `max=255`. It was a lot of experimenting!

Here's an example of my output for this step.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 59 through 67 in the file `helper.py` .  The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32([[270, 700], [540, 500], [1220, 700], [800, 500]])
dst = np.float32([[270, 700], [270, 500], [1180, 700], [1180, 500]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 270, 700     | 270, 700       | 
| 540, 500      | 270, 500      |
| 1220, 700     | 1180, 700      |
| 800, 500      | 1180, 500        |

**Pipeline for waring image:** I started out by drawing four source points on image; two for each vertical line of a lane, like this:
![alt text][image4]

 Then I chose four other points where I wanted to map my warped image to it. I applied `cv2.getPerspectiveTransform` to extract the transformation matrix `M` and unwrap transformation matrix `Minv` i.e. mapping from `dst` to `src` points. finally by using `cv2.warpPerspective I warp the image. result of the warped imaged and corresponding binary image look like this:
 
![alt text][image5]




####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
The code for below steps can be found in `pipeline.py` line 42 through 113.
 
After lane lines are cleanly extracted, by scanning across x axis we can find peaks of accumulated pixels that helps us find lane lines. These peaks  are a good identification of two lines, we can use it to place our starting point of our lines. histogram below visualise what I just explained:  

![alt text][image6]

For finding lines , It is a good idea to narrow down our searches vertical to the starting point images and focus on those area from bottom to top sliding like a window. I created 11 sliding windows with `margin = 80` and `minpx = 20 `, to help set width with +/- margin and to recenter window if needed pixels not found in the rectangle. Then I extracted left and right line pixels While scanning from bottom to top bottom and fit a second order polynomial for each lane like this:     
![alt text][image7]


####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image8]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./tracked_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

