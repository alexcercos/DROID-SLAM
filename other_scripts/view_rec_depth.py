import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str, help="path to image directory")
args = parser.parse_args()

# Change this to your actual reconstruction path
reconstruction_path = args.datapath

# Load the files
tstamps = np.load(f"{reconstruction_path}/tstamps.npy")
images = np.load(f"{reconstruction_path}/images.npy")
disps = np.load(f"{reconstruction_path}/disps.npy")
poses = np.load(f"{reconstruction_path}/poses.npy")
intrinsics = np.load(f"{reconstruction_path}/intrinsics.npy")

num_frames = images.shape[0]
index = 0

norm_max = 10

while True:
    img = images[index].transpose(1, 2, 0).copy()  # (H, W, 3)
    disp = disps[index]

    # Normalize disparity for display
    disp_norm = np.clip(disp, 0, norm_max) / norm_max * 255 #cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    
    disp_color = cv2.applyColorMap(disp_norm.astype(np.uint8), cv2.COLORMAP_JET)

    # Concatenate image and disparity horizontally
    combined = np.hstack((img, disp_color))

    # Show combined image
    cv2.imshow("Image (Left) + Disparity (Right)", combined)

    key = cv2.waitKey(1) & 0xFF

    if cv2.getWindowProperty("Image (Left) + Disparity (Right)", cv2.WND_PROP_VISIBLE) < 1:
        break

    if key == ord('d') or key == 83:  # Right arrow or 'd'
        index = (index + 1) % num_frames
        print(f"Index {index}/{num_frames} ({norm_max:.2f})",end="\r")
    elif key == ord('a') or key == 81:  # Left arrow or 'a'
        index = (index - 1) % num_frames
        print(f"Index {index}/{num_frames} ({norm_max:.2f})",end="\r")
    elif key == ord('z'):
        norm_max*=1.1
        print(f"Index {index}/{num_frames} ({norm_max:.2f})",end="\r")
    elif key == ord('x'):
        norm_max/=1.1
        print(f"Index {index}/{num_frames} ({norm_max:.2f})",end="\r")
    elif key == ord('s'):
        cv2.imwrite("SAVED_DEPTH.png", disp_color)
    elif key == ord('q') or key == 27:  # 'q' or Esc
        break

cv2.destroyAllWindows()