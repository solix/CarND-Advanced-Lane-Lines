import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


#prepare object points and image points for caliberation

objp = np.zeros((6*9,3),np.float32)
#fill up with points from (0,0,0) .....(6,6,0)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)


#Arrays to store object & image points from images
obj_points=[]
img_points=[]

#read caliberation examples with globe
images = glob.glob('camera_cal/calibration*.jpg')



print("finding corners on given images......")
#steps to search corners we iterate through images
for idx, fname in enumerate(images):

	img = cv2.imread(fname)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	#find the corners
	ret , corners = cv2.findChessboardCorners(img,(9,6) ,None)

	#if ret true then draw corners on chessboard and save it to a file
	if ret:
		print('workin on ',fname)
		#append points to array placeholders 
		obj_points.append(objp)
		img_points.append(corners)
		#Draw chessboard corners on original image and save it to a file
		cv2.drawChessboardCorners(gray,(9,6),corners,ret)
		cv2.imwrite('./camera_cal/caliberation_corners'+str(idx)+'.jpg',img)

img = cv2.imread('./camera_cal/caliberation_corners10.jpg')
print("finding corners completed and file is saved")
