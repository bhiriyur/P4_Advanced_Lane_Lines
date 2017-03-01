## Advanced Lane Finding

### Introduction
In this project, we apply some advanced computer vision techniques (primarily implemeted in OpenCV) to identify lanelines from camera images. Specifically, we build a pipeline that reads in a camera image, applies necessary corrections, finds the lane line markings and estimates its curvature, and finally writes a mask on top of the original image to clearly identify the lane extents. All of the steps involved are described below.

The python script that contains the entire implementation is **p4.py**. The main function that implements all the steps involved to overlay lane lines on the image is called **pipeline**. 

[//]: # (Image References)

[image1]: ./output_images/figure_1.png "Camera Calibration"
[image2]: ./output_images/figure_2.png "Undistortion"
[image3]: ./output_images/figure_3.png "Binary thresholding"
[image4]: ./output_images/figure_4.png "Perspective Transform"
[image5]: ./output_images/figure_5.png "Lane Identification"
[image6]: ./output_images/figure_6.png "Unwarp and Overlay"
[image7]: ./output_images/rcurve.png "Curvature and Offset"
[video1]: ./project_video_out.mp4 "Performance"


### 1. Camera Calibration

##### 1.1 Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration is performed in the function **calibrate_camera**. In this function, we read in a series of images from the *camera_cal* folder and identify the associated image points and object points. Using OpenCV functions, the camera camera matrix and distortion coefficients are computed and stored in a pickle file (for future use). The effect of undistortion on a sample image is shown below. 

![alt text][image1]

### 2. Image Pipeline 

#### 2.1 Provide an example of a distortion-corrected image.
The *cv2.undistort* function (as shown in the code block below)is called using the camera matrix and distortion coefficients computed from the previous step.

```python
def undistort_image(img, mtx, dist):
    """Returns an undistorted image given camera matrix and distortion coeffs"""
    return cv2.undistort(img,mtx,dist,None,mtx)
```

Shown below is an example of a distortion corrected image of one of the test images. The difference is not immediately noticable unless we focus on the edges of the image where there are subtle differences. 

![alt text][image2]

#### 2.2 Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

The function **transform_binary** performs the thresholding operations to identify the pixels that correspond to the lane lines. Admittedly this sequence and the associated parameters were based on trial-and-error and seems to work well for most of the test images, though looking at the behaviour on the challenge videos, it looks like this sequence could be improved.

Nevertheless here is the sequence:
- Convert the image into Hue-Saturation-Lightness (since this space works best for images that contain shadows on lane-lines)
- Use a sobel filter in the x-direction and threshold in a high-range to pick lines away from horizontal
- Threshold on s-channel to pick whitish and yellowish pixels
- Based on some a [blog post](https://medium.com/@royhuang_87663/how-to-find-threshold-f05f6b697a00#.9jajirkwl) I read from another student, I used thresholds on all three channels to again pick white/yellow pixels
- **Update**: Now using the color thresholds from HSV space based on another [blog post](https://chatbotslife.com/robust-lane-finding-using-advanced-computer-vision-techniques-46875bb3c8aa?gi=2b00a1f397f)
- Finally I return a binary image that uses bitwise OR to combine all the above computed thresholds

The function is reproduced here:

```python
def transform_binary(img, sobel_kernel=3, s_thresh=(150, 255), sx_thresh=(20, 100)):
    """Canny-like transform to binary image"""

    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel 
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    #sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in y

    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    combined_binary = np.zeros_like(s_binary,np.uint8)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 255

    color_binary = np.dstack((combined_binary,combined_binary,combined_binary))

    # pick from colors
    # hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # yellow = cv2.inRange(hls,(10,0,200),(40,200,255))
    # white =  cv2.inRange(hls,(10,200,150),(40,250,255))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv,(0,100,100),(50,255,255))
    white =  cv2.inRange(hsv,(20,0,180),(255,80,255))
    yw = yellow | white | combined_binary
    yw_color = np.dstack((yw,yw,yw))

    return yw_color #color_binary
```

The result on one of the test images is here:

![alt text][image3]

#### 2.3 Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image

The perspective transformation takes a trapezoidal region in the original image described by the **src** variable below and transforms it to a regular rectangular region identified by the **dst** variable. This converts the image to a bird's eye view perspective.

```python
def perspective_transform(xsize,ysize):
    """Transforms the camera image to top-down view"""
    src = np.float32(
        [[610, 440],
         [670, 440],
         [1041, 678],
         [265, 678]])

    xoff, yoff = 300, 0
    dst = np.float32(
        [[xoff, yoff],
         [xsize-xoff, yoff],
         [xsize-xoff, ysize-yoff],
         [xoff, ysize-yoff]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv
    
# Call Perspective transformation
binary_warped = cv2.warpPerspective(lane_pixels, M, (xsize, ysize), flags=cv2.INTER_LINEAR)
undist_warped = cv2.warpPerspective(undistorted, M, (xsize, ysize), flags=cv2.INTER_LINEAR)
```

The effect of applying this transformation on both the binary thresholded image and the original undistorted image is shown here:

![alt text][image4]

#### 2.4 Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The function **detect_lanelines** is called to compute the quadratic polynomial coefficents corresponding to the lane line pixels that are found. Here a python decorator (*static_vars*) is used to store the computed polynomial fits from the previous frame also. This allows us to reduce the jitter by smoothing out the frame-to-frame computations. 

Another feature implemented here is the averaging of the polynomial coefficients such that both the lane lines (left and right) remain parallel (coefficients of first and second order) and of constant width (coefficient of zero'th order).

**Update**: The function has been modified from the first submission the following ways:
* The windowing routine now skips searching for more pixels once it has reached either the left or the right edge
* The confidence in the polynomial fit increases as the number of non-zero pixels found grows larger.
    - As a corollary, if the number of non-zero pixels found by the windowing routine is low, the confidence in the resulting polynomial fit is also low
    - Using this property, if we have low confidence in the fits, we use the last best fit from a previous frame.
* These changes have resulted in a much smoother behaviour as seen in the video further below.

The corresponding function is reproduced below:

```python
@static_vars(old_left_fit=None, old_right_fit=None, old_nleft=0, old_nright=0)
def detect_lanelines(binary_warped,alpha=1.0,average_curves=True):
    olf = detect_lanelines.old_left_fit
    orf = detect_lanelines.old_right_fit

    # The lane-detection frmo previous curve does not work properly when there is too much deviation
    #left_fit, right_fit, out_img = detect_lanelines_near(binary_warped, olf, orf)

    # Detect using moving windows
    left_fit, right_fit, out_img, nleft, nright = detect_lanelines_swin(binary_warped)

    if olf is None or orf is None:
        olf = left_fit
        orf = right_fit

    pix_threshold1 = 15000
    pix_threshold2 = 10000
    lane_width = 680
    if nleft>pix_threshold1 and nright>pix_threshold1:
        # Very good confidence in both curves
        left_fit[0] = np.average([left_fit[0],right_fit[0]])
        right_fit[0] = left_fit[0]
        left_fit[1] = np.average([left_fit[1],right_fit[1]])
        right_fit[1] = left_fit[1]

        # Don't get lanes shift too much
        center = np.average([left_fit[2],right_fit[2]])
        left_fit[2] = center - lane_width/2.0
        right_fit[2] = center + lane_width/2.0
    elif nleft>pix_threshold2:
        # Good confidence in left curve
        right_fit[:2] = left_fit[:2]
        right_fit[2] = left_fit[2] + lane_width

    elif nright>pix_threshold2:
        # Good confidence in right curve
        left_fit[:2] = right_fit[:2]
        left_fit[2] = right_fit[2] - lane_width

    else:
        # No confidence in either. Use old values
        left_fit = olf
        right_fit = orf

    detect_lanelines.old_left_fit = left_fit
    detect_lanelines.old_right_fit = right_fit
    detect_lanelines.old_nleft = nleft
    detect_lanelines.old_nright = nright


    return left_fit, right_fit, out_img
```

The function that actually computes the polynomial coefficients is **detect_lanelines_swin** which uses a moving window approach that was repreduced directly from the Udacity module content. The following image shows the application of this algorithm and the resulting mask.

![alt text][image5]

#### 2.5 Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Based on the polynomial fit and the left/right lane computation from the previous step, the position of the center of the computed lane was extracted (shown as a red traingular marker on the mask). This was compared to the fixed position of the center of the image (shown as a blue cross mark on the mask). The difference between the above two points was converted from pixel-scale to meter-scale to result in **offset_distance**. This part of the code is implemented in the function **overlay**.

```python
    # Draw offset points
    y0 = ny-5
    x0 = np.int(0.5*(lfit(y0) + rfit(y0)))
    cv2.drawMarker(overlay,(x0,y0),(255,0,0),cv2.MARKER_TRIANGLE_UP,
                  markerSize=10, thickness=1, line_type=cv2.LINE_AA)

    y1 = ny-5
    x1 = np.int(nx/2)+25
    cv2.drawMarker(overlay, (x1, y1), (0, 0, 255), cv2.MARKER_CROSS,
                   markerSize=5, thickness=1, line_type=cv2.LINE_AA)

    yfac = 3.7/700 # Pixel to meters (Y)
    xfac = 100.0/720 # Pixel to meters (X)
    offset_distance = np.round((x0-x1)*yfac,2)
```

To compute the radius of curvature, a set of points that fall on the polynomial fit were first converted to meter-scale and then the polynomial coeffiencts recomputed. Based on these meter-scale coefficients, the radius of curvature was computed in meter-scale corresponding to a point near the bottom of the image (near the current posiiton of the car). The relevant code block is reproduced below:

```python
    # Curvature
    yvals = np.array([ny, 0.75*ny, 0.5*ny])
    xvals = lfit(yvals)

    xnew = xvals*xfac
    ynew = yvals*yfac
    fitnew = np.polyfit(ynew,xnew,2)

    A = fitnew[0]
    B = fitnew[1]
    yeval = y1*yfac/2.0

    rcurve = ((1 + (2*A*yeval+B)**2)**1.5) / np.absolute(2*A)
    rcurve = np.round(rcurve,2)
```

**Update**: I updated the polynomail fit equation (changed independant variable to x) and this resulted in more reasonable values of the radius of curvature. Below image shows a frame with radius of curvature and center-offset displayed:

![alt text][image7]

#### 2.6 Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally the lane mask is undistorted back to the perspective view using the **Minv** computed before and added on top of the original image. Below image shows the final result.

```python
overlaid,overlay1,overlay2,offset_distance,rcurve= overlay_lane(undistorted, left_fit, right_fit, Minv)
```

![alt text][image6]

### 3. Video Pipeline

#### 3.1 Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)

**Revised video**: https://youtu.be/rhONam9MUFc

**Old Version**:
https://youtu.be/ab58qQBpU3k

[![Video](http://img.youtube.com/vi/rhONam9MUFc/0.jpg)](http://www.youtube.com/watch?v=rhONam9MUFc "Advanced Lane Finding")


### 4. Discussion 

#### 4.1 Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

By no means is this project over for me. The performance on the challenge videos was abysmal and forces me to investigate the issues faced by my pipeline. Shadows, change in surface color (yellow on asphalt pavements) and large curvatures seem to pose problems. I plan to improve the following pieces of my pipeline to make it work on the challenge videos.
- Better thresholding of binary mask
- Perhaps the perspetive transformation coordinates need to be reevaluated
- Improved frame-to-frame filtering to smooth wobbliness (should also help the project video)
- Tune the moving-window-selector to identify proper lane pixels.

I will update the [git repository](https://github.com/bhiriyur/P4_Advanced_Lane_Lines) for this project as and when I find the time to make the aforementioned improvements.

