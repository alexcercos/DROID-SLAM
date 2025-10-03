import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--imgpath", type=str, help="path to image directory")
args = parser.parse_args()

img = cv2.imread(args.imgpath, cv2.IMREAD_ANYDEPTH)
img = cv2.merge([img,img,img])
norm_max = 10

while True:

    # Normalize disparity for display
    disp_norm = np.clip(np.where(img>0,1.0 / (img / 5000.0), img), 0, norm_max) / norm_max * 255 #cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)

    disp_color = cv2.applyColorMap(disp_norm.astype(np.uint8), cv2.COLORMAP_PLASMA)

    img2 = img/255.0

    # Concatenate image and disparity horizontally
    combined = np.hstack((img2.astype(np.uint8), disp_color))

    # Show combined image
    cv2.imshow("Depth", combined)

    key = cv2.waitKey(1) & 0xFF

    if cv2.getWindowProperty("Depth", cv2.WND_PROP_VISIBLE) < 1:
        break

    if key == ord('z'):
        norm_max*=1.1
        print(f"({norm_max:.2f})",end="\r")
    elif key == ord('x'):
        norm_max/=1.1
        print(f"({norm_max:.2f})",end="\r")
    elif key == ord('q') or key == 27:  # 'q' or Esc
        break

cv2.destroyAllWindows()