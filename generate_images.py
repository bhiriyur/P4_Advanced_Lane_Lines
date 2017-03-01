from p4 import *
import glob

def plot2(img1,img2,caption1,caption2):
    fig1 = plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.xlabel(caption1)

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.xlabel(caption2)

    return fig1

def draw_points(img,pts):
    mt = cv2.MARKER_SQUARE
    out = np.copy(img)
    for i in range(4):
        pt = tuple(pts[i,:])
        out = cv2.drawMarker(out, pt, (255, 0, 0),
                             markerType=mt, markerSize=10, thickness=4, line_type=cv2.LINE_AA)
    return out


if __name__=='__main__':
    # Distortion correction
    imgfile = 'camera_cal/calibration1.jpg'
    img = plt.imread(imgfile)
    mtx, dist = calibrate_camera()
    undistorted=undistort_image(img, mtx, dist)
    fig1 = plot2(img, undistorted, imgfile, 'undistorted')
    fig1.savefig('output_images/figure_1.png')

    # Distortion correction
    for imgfile in glob.glob('test_images/test3.jpg'):
        #imgfile = 'test_images/test3.jpg'
        img = plt.imread(imgfile)
        mtx, dist = calibrate_camera()
        undistorted=undistort_image(img, mtx, dist)
        fig2 = plot2(img, undistorted, imgfile, 'undistorted')
        fig2.savefig('output_images/figure_2.png')

        # Binary transformation (gradient, sobel operations)
        lane_pixels = transform_binary(undistorted)
        fig3 = plot2(undistorted,lane_pixels,imgfile,'Thresholded/binary')
        fig3.savefig('output_images/figure_3.png')

        # Perspective transformation
        ysize, xsize, nc = img.shape
        M, Minv, src, dst = perspective_transform(xsize, ysize)
        binary_warped = cv2.warpPerspective(lane_pixels, M, (xsize, ysize), flags=cv2.INTER_LINEAR)
        undist_warped = cv2.warpPerspective(undistorted, M, (xsize, ysize), flags=cv2.INTER_LINEAR)
        simg = draw_points(undistorted,src)
        dimg = draw_points(undist_warped,dst)
        fig4 = plot2(simg, dimg, 'Source', 'Destination-Warped')
        #fig4= plot2(binary_warped,undist_warped,'binary_warped','undistorted_warped')
        fig4.savefig('output_images/figure_4.png')

        left_fit, right_fit, win_img = detect_lanelines(binary_warped[:,:,1],alpha=0.9)
        overlaid,overlay1,overlay2,offset_distance,rcurve= overlay_lane(undistorted, left_fit, right_fit, Minv)

        fig5= plot2(win_img,overlay1,'Laneline Identification','Lane mask')
        fig5.savefig('output_images/figure_5.png')

        fig6= plot2(overlay2,overlaid,'Warped back','Overlaid')
        fig6.savefig('output_images/figure_6.png')

        plt.show()
