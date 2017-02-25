import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


def calibrate_camera(plot=False):
    """Calibrate camera based on chessboard images"""

    # Set up object points
    nx = 9
    ny = 6

    objpoints, imgpoints = [], []

    for imgfile in glob.glob('camera_cal/cal*.jpg'):
        # Read image and convert to grayscale
        img = plt.imread(imgfile)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Object points
        objp = np.zeros((nx*ny, 3))
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # Image points
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        print("Calibrating with {:20} returned {}".format(imgfile, ret))

        if ret:
            objpoints.append(objp.astype('float32'))
            imgpoints.append(corners.astype('float32'))

            # Plot if requested
            if plot:
                img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
                plt.imshow(img)
                plt.show()

    # Now calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[:2],None,None)

    return mtx, dist


def undistort_image(img, mtx, dist):
    """Returns an undistorted image given camera matrix and distortion coeffs"""
    return cv2.undistort(img,mtx,dist,None,mtx)

def transform_binary(img, sobel_kernel=3, s_thresh=(150, 255), sx_thresh=(20, 100)):
    """Canny-like transform to binary image"""
    img = np.copy(img)

    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

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

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary

def perspective_transform(img):
    """Transforms the camera image to top-down view"""
    src = np.float32([[550, 480],
           [733, 480],
           [1041, 678],
           [265, 678]])

    xoff, yoff = 300, 10
    ysize, xsize = img.shape
    dst = np.float32([[xoff, yoff],
           [xsize-xoff, yoff],
           [xsize-xoff, ysize-yoff],
           [xoff, ysize-yoff]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (xsize,ysize), flags=cv2.INTER_LINEAR)

    return warped, M, Minv

def detect_lanelines_near(binary_warped,left_fit, right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def detect_lanelines_swin(binary_warped):
    """This part of the code was based on Udacity module content"""
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
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
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
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

    return left_fit, right_fit

def pipeline(img,mtx=None,dist=None,plot=False):
    """
    Perform the following steps in sequence to find lanes
    * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    * Apply a distortion correction to raw images.
    * Use color transforms, gradients, etc., to create a thresholded binary image.
    * Apply a perspective transform to rectify binary image ("birds-eye view").
    * Detect lane pixels and fit to find the lane boundary.
    * Determine the curvature of the lane and vehicle position with respect to center.
    * Warp the detected lane boundaries back onto the original image.
    * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    """
    if mtx is None or dist is None:
        mtx, dist = calibrate_camera()

    undistorted=undistort_image(img, mtx, dist)
    sbin = transform_binary(undistorted)
    binary_warped, M, Minv = perspective_transform(sbin)
    left_fit, right_fit = detect_lanelines_swin(binary_warped)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Generate an overlay from left and right lanes
    n1 = len(left_fitx)
    n2 = len(right_fitx)
    polypoints = np.zeros((n1+n2,2))
    polypoints[:n1, 0] = left_fitx
    polypoints[:n1, 1] = ploty
    polypoints[n1:, 0] = right_fitx[-1::-1]
    polypoints[n1:, 1] = ploty[-1::-1]

    overlay = np.zeros_like(undistorted)
    output = undistorted.copy()

    cv2.fillPoly(overlay, np.int32([polypoints]),(0, 255, 0))
    overlay = cv2.warpPerspective(overlay, Minv, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
    # apply the overlay
    alpha = 0.5
    dst = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

    nx, ny = 2,2
    if plot:
        plt.figure(figsize=(10,8))
        plt.subplot(nx,ny,1)
        plt.imshow(img)

        plt.subplot(nx,ny,2)
        plt.imshow(undistorted)      

        plt.subplot(nx,ny,3)
        plt.imshow(sbin,cmap='gray')

        plt.subplot(nx,ny,4)
        plt.imshow(dst)
        #plt.plot(polypoints[:,0],polypoints[:,1],color='yellow',lw=4)
        #plt.plot(right_fitx, ploty, color='yellow', lw=4)
        plt.show()


    #detectLanes()
    #computeCurvature()
    #warpBack()
    #outputCurvature()
    return


if __name__=='__main__':
    """Main thread entry"""

    mtx, dist = calibrate_camera()

    for test_image_file in glob.glob('test_images/*.jpg'):
        test_image = plt.imread(test_image_file)
        pipeline(test_image, mtx, dist, plot=True)
