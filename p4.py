import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from moviepy.editor import VideoFileClip
import pickle
import os.path

def calibrate_camera(plot=False):
    """Calibrate camera based on chessboard images"""

    # See if we have stored calibration info previously
    calfile = 'camera_cal.pkl'
    if os.path.isfile(calfile):
        data = pickle.load(open(calfile, 'rb'))
        return data[0],data[1]

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

    # Store in calfile for future
    data = [mtx,dist]
    pickle.dump(data, open(calfile, 'wb'))

    return mtx, dist


def undistort_image(img, mtx, dist):
    """Returns an undistorted image given camera matrix and distortion coeffs"""
    return cv2.undistort(img,mtx,dist,None,mtx)

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

def perspective_transform(xsize,ysize):
    """Transforms the camera image to top-down view"""
    src = np.float32(
        [[610, 440],
         [670, 440],
         [1041, 678],
         [265, 678]])

    # x1 = 515
    # x2 = 10
    # y1 = 480
    # src = np.float32(
    #     [[x1, y1],
    #      [xsize-x1, y1],
    #      [xsize-x2, ysize],
    #      [x2, ysize]])



    xoff, yoff = 300, 0
    dst = np.float32(
        [[xoff, yoff],
         [xsize-xoff, yoff],
         [xsize-xoff, ysize-yoff],
         [xoff, ysize-yoff]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

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
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    return left_fit, right_fit, out_img

def detect_lanelines_swin(binary_warped):
    """This part of the code was based on Udacity module content"""
    # Assuming you have created a warped binary image called "binary_warped"

    ny, nx = binary_warped.shape

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(0.75*ny):, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) #* 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Space out base points
    if abs((rightx_base-leftx_base)-680) > 50:
        rightx_base = leftx_base + 680

    # Choose the number of sliding windows
    nwindows = 11

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
    count, lcount, rcount = 0, 0, 0
    for window in range(nwindows):
        count += 1
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
        good_left_inds = ((nonzeroy >= win_y_low) &
                          (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) &
                           (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        lcount += len(good_left_inds)
        rcount += len(good_right_inds)
        #print("Window #{}, Lpixels = {}, Rpixels = {}".format(count,lcount,rcount))
        #if lcount>20000 or rcount>20000:
        #    break
        if win_xleft_low<=5 or win_xright_high>=(nx-5):
            break

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        #print(left_fit,right_fit)
    except TypeError:
        left_fit = [0.0, 0.0, -200.0]
        right_fit = [0.0, 0.0, 450.0]

    return left_fit, right_fit, out_img, len(leftx), len(lefty)

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

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

def overlay_lane(undistorted, left_fit, right_fit, Minv):

    ny, nx, nc = undistorted.shape

    lfit = lambda y: left_fit[0] * (y**2) + left_fit[1] * y + left_fit[2]
    rfit = lambda y: right_fit[0] * (y**2) + right_fit[1] * y + right_fit[2]

    # Generate x and y values for plotting
    ybar = np.linspace(0, undistorted.shape[0] - 1, undistorted.shape[0])
    lxbar = lfit(ybar)
    rxbar = rfit(ybar)

    # Generate an overlay from left and right lanes
    n = len(ybar)
    polypoints = np.zeros((2*n, 2))
    polypoints[:n, 0] = lxbar
    polypoints[:n, 1] = ybar
    polypoints[n:, 0] = rxbar[-1::-1]
    polypoints[n:, 1] = ybar[-1::-1]

    overlay = np.zeros_like(undistorted)
    output = undistorted.copy()

    cv2.fillPoly(overlay, np.int32([polypoints]), (0, 255, 0))

    # Draw lane lines
    for i in range(n-1):
        x1,y1 = np.int(lxbar[i]), np.int(ybar[i])
        x2,y2 = np.int(lxbar[i+1]),np.int(ybar[i+1])
        cv2.line(overlay, (x1, y1), (x2, y2),(255,255,255),30)
        x1,y1 = np.int(rxbar[i]), np.int(ybar[i])
        x2,y2 = np.int(rxbar[i+1]),np.int(ybar[i+1])
        cv2.line(overlay, (x1, y1), (x2, y2),(255,255,255),30)

    # Draw offset points
    y0 = ny-5
    x0 = np.int(0.5*(lfit(y0) + rfit(y0)))
    cv2.drawMarker(overlay,(x0,y0),(255,0,0),cv2.MARKER_TRIANGLE_UP,
                  markerSize=10, thickness=1, line_type=cv2.LINE_AA)

    y1 = ny-5
    x1 = np.int(nx/2)+25
    cv2.drawMarker(overlay, (x1, y1), (0, 0, 255), cv2.MARKER_CROSS,
                   markerSize=5, thickness=1, line_type=cv2.LINE_AA)

    xfac = 3.7/700 # Pixel to meters (Y)
    yfac = 100.0/720 # Pixel to meters (X)
    offset_distance = np.round((x0-x1)*xfac,2)

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
    overlay_unwarp = cv2.warpPerspective(overlay, Minv, (nx, ny), flags=cv2.INTER_LINEAR)

    # apply the overlay
    overlaid=cv2.addWeighted(output, 1, overlay_unwarp, 0.3, 0)

    return overlaid,overlay,overlay_unwarp, offset_distance, rcurve

@static_vars(old_rcurve=None)
def pipeline(img,mtx=None,dist=None,M=None,Minv=None,plot=False):
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
    # Get size of image
    ysize, xsize, nc = img.shape

    # Calibrate camera
    if mtx is None or dist is None:
        mtx, dist = calibrate_camera()

    if M is None or Minv is None:
        M, Minv = perspective_transform(xsize, ysize)

    # Undistort camera
    undistorted=undistort_image(img, mtx, dist)

    # Binary transformation (gradient, sobel operations)
    lane_pixels = transform_binary(undistorted)

    # Perspective transformation
    binary_warped = cv2.warpPerspective(lane_pixels, M, (xsize, ysize), flags=cv2.INTER_LINEAR)
    undist_warped = cv2.warpPerspective(undistorted, M, (xsize, ysize), flags=cv2.INTER_LINEAR)

    left_fit, right_fit, win_img = detect_lanelines(binary_warped[:,:,0])
    overlaid,overlay1,overlay2,offset_distance,rcurve= overlay_lane(undistorted, left_fit, right_fit, Minv)

    #offset_distance,rcurve = get_offset_rcurve(left_fit,right_fit,xsize,ysize)

    # save in static vars
    if pipeline.old_rcurve is None:
        orc = rcurve
    else:
        orc = pipeline.old_rcurve

    # if rcurve<100:
    #     rcurve = -100.0
    # elif rcurve<orc and abs(rcurve-orc)/orc > 0.2:
    #     rcurve = orc
    # else:
    #     pipeline.old_rcurve = rcurve



    out = diag_screen(overlaid, undistorted, undist_warped, lane_pixels, binary_warped, win_img, overlay1,overlay2,offset_distance,rcurve)

    return out

def process_video(vidfile,writeToFile=True):
    mtx, dist = calibrate_camera()
    lanevid = vidfile.split('.')[0] + '_out.mp4'
    clip1 = VideoFileClip(vidfile)
    vpipeline = lambda x: pipeline(x,mtx,dist)
    white_clip = clip1.fl_image(vpipeline)
    #white_clip.preview(fps=25)
    if writeToFile: white_clip.write_videofile(lanevid, audio=False)
    return lanevid


def diag_screen(mainDiagScreen,diag1,diag2,diag3,diag4,diag5,diag6,diag7,offset=0,curvature=0):
    # middle panel text example
    # using cv2 for drawing text in diagnostic pipeline.
    font = cv2.FONT_HERSHEY_PLAIN
    middlepanel = np.zeros((320, 240, 3), dtype=np.uint8)
    cv2.putText(middlepanel, 'Curvature:{} m.'.format(curvature), (20, 60), font, 0.5, (255,255,255), 1)
    cv2.putText(middlepanel, 'Offset:{} m.'.format(offset), (20, 90), font, 0.5, (255,255,255), 1)

    cv2.putText(mainDiagScreen, 'Curvature:{} m.'.format(curvature), (20, 60), font, 2, (255,255,255), 1)
    cv2.putText(mainDiagScreen, 'Offset:{} m.'.format(offset), (20, 90), font, 2, (255,255,255), 1)

    # assemble the screen
    diagScreen = np.zeros((960, 1600, 3),np.uint8)
    diagScreen[0:720, 0:1280] = mainDiagScreen

    # Right screens
    diagScreen[0:240, 1280:1600] = cv2.resize(diag1, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[240:480, 1280:1600] = cv2.resize(diag2, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[480:720, 1280:1600] = cv2.resize(middlepanel, (320, 240), interpolation=cv2.INTER_AREA)

    # Bottom screens
    diagScreen[720:960, 0:320] = cv2.resize(diag3, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[720:960, 320:640] = cv2.resize(diag4, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[720:960, 640:960] = cv2.resize(diag5, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[720:960, 960:1280] = cv2.resize(diag6, (320, 240), interpolation=cv2.INTER_AREA)
    diagScreen[720:960, 1280:1600] = cv2.resize(diag7, (320, 240), interpolation=cv2.INTER_AREA)



    return diagScreen



if __name__=='__main__':
    """Main thread entry"""
    vidfile = 'project_video.mp4'
    lanevid = process_video(vidfile)
    print("saved to {}".format(lanevid))

    # mtx, dist = calibrate_camera()
    #
    # for test_image_file in glob.glob('test_images/*.jpg'):
    #     test_image = cv2.imread(test_image_file)
    #     screen = pipeline(test_image, mtx, dist)
    #     cv2.imshow('Diagnostic Screen',screen)
    #     cv2.waitKey()
