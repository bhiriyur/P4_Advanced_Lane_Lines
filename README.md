## Advanced Lane Finding

### Introduction
In this project, we apply some advanced computer vision techniques (primarily implemeted in OpenCV) to identify lanelines from camera images. Specifically, we build a pipeline that reads in a camera image, applies necessary corrections, finds the lane line markings and estimates its curvature, and finally writes a mask on top of the original image to clearly identify the lane extents. All of the steps involved are described below.

The python script that contains the entire implementation is **p4.py**.

### 1. Camera Calibration

##### 1.1 Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration is performed in the function **calibrate_camera**. In this function, we read in a series of images from the folder *camera_cal* and identify the associated image points and object points. Using OpenCV functions, the camera camera matrix and distortion coefficients are computed and stored in a pickle file (for future use). The effect of undistortion on a sample image is shown below:
 
 

### 2. Image Pipeline 

##### 2.1 Provide an example of a distortion-corrected image.
##### 2.2 Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.
##### 2.3 Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image
##### 2.4 Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
##### 2.5 Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
##### 2.6 Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

### 3. Video Pipeline

##### 3.1 Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)

### 4. Discussion 

##### 4.1 Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?




