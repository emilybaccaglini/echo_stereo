import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 

np.set_printoptions(suppress=True)
# Object points in 3D
GRID_SHAPE = (9,6)
objp = np.zeros((GRID_SHAPE[0]*GRID_SHAPE[1],3), np.float32)
objp[:,:2] = np.mgrid[0:GRID_SHAPE[0], 0:GRID_SHAPE[1]].T.reshape(-1,2)
objp *= 27 # One square on my grid has 20mm
FOLDER = "calib_images2/left/"
fnames = os.listdir(FOLDER)
print(fnames)
obj_pts = []
img_pts = []


for fname in fnames:
    print(f"processing {fname}")
    img = cv2.imread(os.path.join(FOLDER, fname))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    arr = np.array(gray)
    
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(arr,GRID_SHAPE,flags)
    
    #rr_vis = cv2.drawChessboardCorners(arr, GRID_SHAPE, corners, ret)
    #plt.imshow(arr_vis, cmap='gray')
    #plt.show()

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(arr,corners,(11,11),(-1,-1),criteria)
        obj_pts.append(objp)
        img_pts.append(corners_subpix)

ret, K_l, dist_coeff_l, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (arr.shape[1], arr.shape[0]), None,None)


np.set_printoptions(suppress=True)
# Object points in 3D
GRID_SHAPE = (9,6)
objp = np.zeros((GRID_SHAPE[0]*GRID_SHAPE[1],3), np.float32)
objp[:,:2] = np.mgrid[0:GRID_SHAPE[0], 0:GRID_SHAPE[1]].T.reshape(-1,2)
objp *= 27 # One square on my grid has 20mm
FOLDER = "calib_images2/right/"
fnames = os.listdir(FOLDER)
obj_pts = []
img_pts = []


for fname in fnames:
    print(f"processing {fname}")
    img = cv2.imread(os.path.join(FOLDER, fname))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    arr = np.array(gray)
    
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(arr,GRID_SHAPE,flags)
    
    #rr_vis = cv2.drawChessboardCorners(arr, GRID_SHAPE, corners, ret)
    #plt.imshow(arr_vis, cmap='gray')
    #plt.show()

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(arr,corners,(11,11),(-1,-1),criteria)
        obj_pts.append(objp)
        img_pts.append(corners_subpix)

ret, K_r,dist_coeff_r, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (arr.shape[1], arr.shape[0]), None,None)

np.save("K_l.npy", K_l)
np.save("K_r.npy", K_r)

np.save("dist_coeff_l.npy", dist_coeff_l)
np.save("dist_coeff_r.npy", dist_coeff_r)



# Object points in 3D
GRID_SHAPE = (9,6)
objp = np.zeros((GRID_SHAPE[0]*GRID_SHAPE[1],3), np.float32)
objp[:,:2] = np.mgrid[0:GRID_SHAPE[0], 0:GRID_SHAPE[1]].T.reshape(-1,2)
objp *= 27 # One square on my grid has 20mm

folder_right = "calib_images2/right/"
folder_left = "calib_images2/left/"

# Load parameters
(K_l, K_r, dist_l, dist_r) = np.load("K_l.npy"), np.load("K_r.npy"), np.load("dist_coeff_l.npy"), np.load("dist_coeff_r.npy")

obj_pts = []
img_pts_l, img_pts_r = [], []

for fname_l, fname_r in zip(sorted(os.listdir(folder_left)), sorted(os.listdir(folder_right))):

    print(f"processing {fname_l, fname_r}")
    img_l, img_r = cv2.imread(os.path.join(folder_left,fname_l)),cv2.imread(os.path.join(folder_right,fname_r))
    grey_l,grey_r = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY),cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)
    arr_l, arr_r = np.array(grey_l),np.array(grey_r)
    gray_l, gray_r  = arr_l, arr_r # the images are already grayscale so no conversion

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, GRID_SHAPE, flags)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, GRID_SHAPE, flags)

    if ret_l and ret_r:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)
        corners_subpix_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria)

        obj_pts.append(objp)
        img_pts_l.append(corners_subpix_l)
        img_pts_r.append(corners_subpix_r)

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retS, K_l, dist_l, K_r, dist_r, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(
    obj_pts,
    img_pts_l,
    img_pts_r,
    K_l, dist_l,
    K_r, dist_r,
    gray_l.shape[::-1],
    criteria_stereo,
    flags)

rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(K_l, dist_l, K_r, dist_r, gray_l.shape[::-1], Rot, Trns, 1 ,(0,0))

left_stereo_maps = cv2.initUndistortRectifyMap(K_l, dist_l, rect_l, proj_mat_l,
                                             gray_l.shape[::-1], cv2.CV_16SC2)
right_stereo_maps = cv2.initUndistortRectifyMap(K_r, dist_r, rect_r, proj_mat_r,
                                              gray_l.shape[::-1], cv2.CV_16SC2)


arr_l_rect = cv2.remap(arr_l, left_stereo_maps[0],left_stereo_maps[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
arr_r_rect = cv2.remap(arr_r, right_stereo_maps[0],right_stereo_maps[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)


cv_file = cv2.FileStorage("rectify_map_imx219_160deg_1080p_new.yaml", cv2.FILE_STORAGE_WRITE)
cv_file.write("map_l_1", left_stereo_maps[0])
cv_file.write("map_l_2", left_stereo_maps[1])
cv_file.write("map_r_1", right_stereo_maps[0])
cv_file.write("map_r_2", right_stereo_maps[1])
cv_file.release() 

fs = cv2.FileStorage("rectify_map_imx219_160deg_1080p_new.yaml", cv2.FILE_STORAGE_READ)

map_l = (
    fs.getNode('map_l_1').mat(),
    fs.getNode('map_l_2').mat()
)

map_r = (
    fs.getNode('map_r_1').mat(),
    fs.getNode('map_r_2').mat()
)

fs.release()





