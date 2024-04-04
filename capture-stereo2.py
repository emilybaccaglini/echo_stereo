import time
import cv2
import sys
import numpy as np
import threading

sys.path.append("..")
from start_cameras import Start_Cameras  #

if __name__ == "__main__":

    left_camera = Start_Cameras(0).start()
    right_camera = Start_Cameras(1).start()
    cnt = 0

    try:
        while True:

            print("Capturing. Press spacebar to capture an image pair.")

            left_grabbed, left_img = left_camera.read()
            right_grabbed, right_img = right_camera.read()

            if left_grabbed and right_grabbed:
                img_lr = np.hstack([left_img, right_img])
                img_lr = cv2.resize(img_lr, (1920, 1080//2))

                cv2.imshow("windowname", img_lr)

                key = cv2.waitKey(1) & 0xFF

                if key == ord(" "):
                    cnt += 1
                    cv2.imwrite(f"/home/emily/jetson-stereo-depth/calib/calib_images2/right/{cnt:03d}.png", right_img)
                    cv2.imwrite(f"/home/emily/jetson-stereo-depth/calib/calib_images2/left/{cnt:03d}.png", left_img)
                    print("Image pair captured!")

                if key == ord("q"):
                    break
            else:
                print("Failed to capture images.")
                break
    except KeyboardInterrupt as e:
        print("Closing")
    finally:
        left_camera.stop()
        left_camera.release()
        right_camera.stop()
        right_camera.release()
        cv2.destroyAllWindows()
