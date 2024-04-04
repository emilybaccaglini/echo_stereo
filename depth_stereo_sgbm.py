import time
import queue
import numpy as np
import cv2
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from start_cameras import Start_Cameras  # Import the Start_Cameras class

# Adjust these parameters according to your requirements
WINDOW_SIZE = 5  # Adjust window size if needed
MIN_DISP = 16
MAX_DISP = 128

class Depth(Thread):
    def __init__(self):
        super().__init__()
        print("Reading camera calibration...")
        self._map_l, self._map_r = get_calibration()
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
        i = 0
        while self._should_run:
            i += 1
            with cv2.cuda.GpuMat() as gpu_left, cv2.cuda.GpuMat() as gpu_right:
                # Read Images
                left_grabbed, arr_l = self._left_camera.read()
                right_grabbed, arr_r = self._right_camera.read()

                if left_grabbed and right_grabbed:
                    # Upload images to GPU
                    gpu_left.upload(arr_l)
                    gpu_right.upload(arr_r)

                    # Convert to grayscale
                    gpu_left_gray = cv2.cuda.cvtColor(gpu_left, cv2.COLOR_BGR2GRAY)
                    gpu_right_gray = cv2.cuda.cvtColor(gpu_right, cv2.COLOR_BGR2GRAY)

                    # Compute disparity map using StereoSGBM
                    sgbm = cv2.cuda.createStereoSGBM(minDisparity=MIN_DISP, numDisparities=MAX_DISP, blockSize=WINDOW_SIZE)
                    gpu_disp = sgbm.compute(gpu_left_gray, gpu_right_gray)

                    # Download disparity map from GPU
                    disp_arr = gpu_disp.download()
                    self._executor.submit(self.enqueue_async, disp_arr)

def get_calibration() -> tuple:
    fs = cv2.FileStorage("/home/emily/jetson-stereo-depth/calib/rectify_map_imx219_160deg_1080p_new.yaml", cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("Calibration file not found")
    map_l = (fs.getNode("map_l_1").mat(), fs.getNode("map_l_2").mat())
    map_r = (fs.getNode("map_r_1").mat(), fs.getNode("map_r_2").mat())
    fs.release()
    return map_l, map_r


if __name__ == "__main__":
    DISPLAY = True
    depth = Depth()
    t1 = time.perf_counter()

    try:
        while True:
            disp_arr = depth.disparity()
            if DISPLAY:
                disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_JET)
                cv2.imshow("Depth", disp_arr)
                left_grabbed, left_frame = depth._left_camera.read()
                cv2.imshow("Image", cv2.resize(left_frame, (480, 270)))
                key = cv2.waitKey(1)
                if key and 0xFF == ord('q'):
                    break

    finally:
        depth.stop()
        t2 = time.perf_counter()




