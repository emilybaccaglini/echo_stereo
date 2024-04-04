import time
import queue
import numpy as np
import cv2
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from start_cameras import Start_Cameras  # Import the Start_Cameras class

MAX_DISP = 32
WINDOW_SIZE = 1000

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
        import vpi  # Don't import vpi in main thread to avoid creating a global context

        i = 0
        self._warp_l = make_vpi_warpmap(self._map_l)
        self._warp_r = make_vpi_warpmap(self._map_r)
        ts_history = []

        while self._should_run:
            i += 1
            ts = []
            ts.append(time.perf_counter())

            with vpi.Backend.CUDA:
                ts.append(time.perf_counter())

                # Read Images
                left_grabbed, arr_l = self._left_camera.read()
                right_grabbed, arr_r = self._right_camera.read()

                if left_grabbed and right_grabbed:
                    ts.append(time.perf_counter())

                    # Convert to VPI image
                    vpi_l = vpi.asimage(arr_l)
                    vpi_r = vpi.asimage(arr_r)
                    ts.append(time.perf_counter())

                    # Rectify
                    vpi_l = vpi_l.remap(self._warp_l)
                    vpi_r = vpi_r.remap(self._warp_r)
                    ts.append(time.perf_counter())
                    vpi_l = vpi_l.convert(vpi.Format.U8, scale=1)
                    vpi_r = vpi_r.convert(vpi.Format.U8, scale=1)
                    # Resize
                    vpi_l = vpi_l.rescale((480, 270), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
                    vpi_r = vpi_r.rescale((480, 270), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
                    ts.append(time.perf_counter())

                    # Convert to 16bpp
                    vpi_l_16bpp = vpi_l.convert(vpi.Format.U16, scale=1)
                    vpi_r_16bpp = vpi_r.convert(vpi.Format.U16, scale=1)
                    ts.append(time.perf_counter())

                    # Disparity
                    disparity_16bpp = vpi.stereodisp(
                        vpi_l_16bpp,
                        vpi_r_16bpp,
                        out_confmap=None,
                        backend=vpi.Backend.CUDA,
                        window=WINDOW_SIZE,
                        maxdisp=MAX_DISP,
                    )
                    ts.append(time.perf_counter())

                    # Convert again
                    disparity_8bpp = disparity_16bpp.convert(vpi.Format.U8, scale=255.0 / (32 * MAX_DISP))
                    ts.append(time.perf_counter())

                    # CPU mapping
                    self._executor.submit(self.enqueue_async, disparity_8bpp.cpu())
                    ts.append(time.perf_counter())

                    ts.append(time.perf_counter())

                    ts = np.array(ts)
                    ts_deltas = np.diff(ts)

                    ts_history.append(ts_deltas)

                    if i % 10 == 0:
                        vpi.clear_cache()

def disp2depth(disp_arr):
    F, B, PIXEL_SIZE = (CAM_PARAMS.F, CAM_PARAMS.B, CAM_PARAMS.PIXEL_SIZE)
    disp_arr[disp_arr < 10] = 10
    disp_arr_mm = disp_arr * PIXEL_SIZE  # convert disparites from px -> mm
    depth_arr_mm = F * B / (disp_arr_mm + 1e-10)  # calculate depth
    return depth_arr_mm

def get_calibration() -> tuple:
    fs = cv2.FileStorage("/home/emily/jetson-stereo-depth/calib/rectify_map_imx219_160deg_1080p_new.yaml", cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("Calibration file not found")
    map_l = (fs.getNode("map_l_1").mat(), fs.getNode("map_l_2").mat())
    map_r = (fs.getNode("map_r_1").mat(), fs.getNode("map_r_2").mat())
    fs.release()
    return map_l, map_r

def make_vpi_warpmap(cv_maps):
    import vpi

    src_map = cv_maps[0]
    idk_what_that_is = cv_maps[1]
    map_y, map_x = src_map[:, :, 0], src_map[:, :, 1]

    warp = vpi.WarpMap(vpi.WarpGrid((1920, 1080)))
    # (H, W, C) -> (C,H,W)
    arr_warp = np.asarray(warp)
    wx = arr_warp[:, :, 0]
    wy = arr_warp[:, :, 1]

    wy[:1080, :] = map_x
    wx[:1080, :] = map_y

    return warp

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
        #print(f"Approx framerate: {len(frames_d)/(t2-t1)} FPS")
        
