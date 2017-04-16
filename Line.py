import numpy as np
import cv2
import collections
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = collections.deque([],maxlen=10)
        #average x values of the fitted line over the last n=10 iterations
        self.bestx = np.mean(self.recent_xfitted, axis=0)
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = collections.deque([],maxlen=20)
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.prev_fit = None
       

    
