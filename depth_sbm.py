import time
import queue
import numpy as np
import cv2
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from stereovision.calibration import StereoCalibration
from start_cameras import Start_Cameras  # Import the Start_Cameras class

MAX_DISP = 64
WINDOW_SIZE = 5
MIN_DISP = 0
P1=8
P2=32
SWS=20
SR= 2
UR=1
PFC=0
D12MD=0
class Depth(Thread):
    def __init__(self):
        super().__init__()
        print("Reading camera calibration...")
        self._map_l_1, self._map_r_1, self._map_l_2, self._map_r_2 = get_calibration()
        self._disp_arr = None
        self._should_run = True
        self._dq = queue.deque(maxlen=3)
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._left_camera = Start_Cameras(0).start()  # Initialize left camera
        self._right_camera = Start_Cameras(1).start()  # Initialize right camera
        self.start()

        # Wait for the deque to start filling up
        while len(self._dq) < 1:
            time.sleep(0.1)

    def stop(self):
        self._should_run = False
        self._left_camera.stop()
        self._right_camera.stop()

    def disparity(self):
        while len(self._dq) == 0:
            time.sleep(0.01)
        return self._dq.pop()

    def enqueue_async(self, disp_arr):
        self._dq.append(disp_arr)

    def run(self):

     
        
        ts_history = []

        while self._should_run:
            
            ts = []
            ts.append(time.perf_counter())

           
            

            # Read Images
            left_grabbed, arr_l = self._left_camera.read()
            right_grabbed, arr_r = self._right_camera.read()

            if left_grabbed and right_grabbed:
                ts.append(time.perf_counter())

                # Rectify
                calibration = StereoCalibration(input_folder = '/home/jetson/StereoVision/calib_result')
                arr_l, arr_r = calibration.rectify((arr_l, arr_r))
                #arr_l = cv2.remap(arr_l, self._map_l_1, self._map_l_2, cv2.INTER_LINEAR)
                #arr_r = cv2.remap(arr_r, self._map_r_1, self._map_r_2, cv2.INTER_LINEAR)
                ts.append(time.perf_counter())
                # Resize
                gray_l = cv2.resize(cv2.cvtColor(arr_l, cv2.COLOR_BGR2GRAY),(480,270))
                gray_r = cv2.resize(cv2.cvtColor(arr_r, cv2.COLOR_BGR2GRAY),(480,270))
                ts.append(time.perf_counter())

                #print("resized")
                # Disparity
                sbm = cv2.StereoSGBM_create(numDisparities=MAX_DISP, blockSize=WINDOW_SIZE, minDisparity=MIN_DISP,P1=P1,P2=P2,speckleWindowSize=SWS,speckleRange=SR,disp12MaxDiff=D12MD,uniquenessRatio=UR,preFilterCap=PFC)
                #match = cv2.ximgproc.createRightMatcher(sbm)
                disp_arr = sbm.compute(gray_l, gray_r)
                #disp_match = match.compute(gray_r, gray_l)
                ts.append(time.perf_counter())
                
                # normalize and Convert
                disp_arr = cv2.normalize(disp_arr, None, 0, 255, cv2.NORM_MINMAX)
                #disp_arr = np.array(disp_arr, dtype = np.uint8)
                #colormap = custom_rainbow_colormap()
                #disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_JET)
                #disp_arr = cv2.bitwise_not(disp_arr)
                ts.append(time.perf_counter())
                #filter
                #wls = cv2.ximgproc.createDisparityWLSFilter(sbm)
                #wls.setLambda(8000)
                #wls.setSigmaColor(1.5)
                #disp_arr = wls.filter(disp_arr, gray_l, disparity_map_right=disp_match)
                #disp_arr = cv2.normalize(disp_arr, None, 0, 255, cv2.NORM_MINMAX)
                #disp_arr = np.array(disp_arr, dtype = np.uint8)
                #disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_JET)
                ts.append(time.perf_counter())
                # CPU mapping
                self._executor.submit(self.enqueue_async, disp_arr)
                #("mapped")
                ts.append(time.perf_counter())

                ts.append(time.perf_counter())

                ts = np.array(ts)
                ts_deltas = np.diff(ts)

                ts_history.append(ts_deltas)
                

                    
class PARAMS:
    SENSOR_FULL_X = 3.68 # mm
    SENSOR_FULL_Y = 2.76 # mm
    SENSOR_FULL_X_PX = 3280
    SENSOR_FULL_Y_PX = 2464

    PIXEL_SIZE = SENSOR_FULL_X/3280 # 1.12 um for IMX219
    F_PX = 1570 # focal length in pixels
    F = F_PX*PIXEL_SIZE
    B = 100  # mm (baseline)
def disp2depth(disp_arr):
    disp_arr_cm = int(265559700 +(51.505-265559700)/(1+(disp_arr/2449.329)**5.262))
    return disp_arr_cm
def onMouse(event, x,y, flag, disp_arr):
    if event == cv2.EVENT_LBUTTONDOWN:
        distance = disp2depth(disp_arr[y][x])
        #r, g, b = distance
        #distance = 0.2989*r + 0.5870*g + 0.1140*b
        #distance = disp2depth(distance)	
        print("Distance in cm {}".format(distance))
        return distance
def get_calibration() -> tuple:
    fs = cv2.FileStorage("/home/jetson/echo_stereo/calib/rectify_map_imx219_160deg_1080p_new.yaml", cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("Calibration file not found")
    map_l_1, map_l_2 = fs.getNode("map_l_1").mat(), fs.getNode("map_l_2").mat()
    map_r_1, map_r_2 = fs.getNode("map_r_1").mat(), fs.getNode("map_r_2").mat()
    fs.release()
    return map_l_1, map_r_1, map_l_2, map_r_2

import cv2
import numpy as np

def custom_rainbow_colormap():
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)

    # Assigning dark blue (only blue channel to max) for values 0-5
    for i in range(200):
        colormap[i, 0, 0] = 0    # Red channel
        colormap[i, 0, 1] = 0    # Green channel
        colormap[i, 0, 2] = 127  # Blue channel

    # Rest of the colormap follows the rainbow pattern
    for i in range(6, 256):
        # The color values follow a gradient from blue to green to red
        colormap[i, 0, 0] = max(0, min(4 * (i - 128), 255))  # Red channel
        colormap[i, 0, 1] = max(0, min(512 - 4 * abs(i - 128), 255))  # Green channel
        colormap[i, 0, 2] = max(0, min(255 - 4 * (i - 128), 255))  # Blue channel

    return colormap


if __name__ == "__main__":
    #DISPLAY = True
    #SAVE = False
    #frames_d = []
    #frames_rgb = []

    #depth = Depth()
    #t1 = time.perf_counter()

    #for i in range(50):
        #disp_arr = depth.disparity()
        #frames_d.append(disp_arr)
        #left_grabbed, left_frame = depth._left_camera.read()
        #frames_rgb.append(left_frame)
        #print(i)

        #if DISPLAY:
            #disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_JET)
            #cv2.imshow("Depth", disp_arr)
            #cv2.imshow("Image", cv2.resize(left_frame, (480, 270)))
            #cv2.waitKey(1)

    #depth.stop()
    #t2 = time.perf_counter()
    #print(f"Approx framerate: {len(frames_d)/(t2-t1)} FPS")

    # Save frames
    #for i, (disp_arr, rgb_arr) in enumerate(zip(frames_d, frames_rgb)):
        #print(f"{i}/{len(frames_d)}", end="\r")
        #disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_JET)
        #cv2.imwrite(f"/home/emily/jetson-stereo-depth/save_frames/depth_{i}.jpg", disp_arr)
        #cv2.imwrite(f"/home/emily/jetson-stereo-depth/save_frames/rgb_{i}.jpg", rgb_arr)


    DISPLAY = True
    depth = Depth()
    t1 = time.perf_counter()

    try:
        while True:
            disp_arr = depth.disparity()
            if DISPLAY:
                cv2.setMouseCallback("Depth", onMouse, disp_arr)
                disp_arr = np.array(disp_arr, dtype = np.uint8)
                disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_RAINBOW)
                cv2.imshow("Depth", disp_arr)
                left_grabbed, left_frame = depth._left_camera.read()
                cv2.imshow("Image", cv2.resize(left_frame, (480, 270)))
                key = cv2.waitKey(1)
                if key and 0xFF == ord('q'):
                    break

    finally: 
        depth.stop()
        t2 = time.perf_counter()
        #print(f"Approx framerate: {len(frames_d)/(t2-t1)} FPS")
        
