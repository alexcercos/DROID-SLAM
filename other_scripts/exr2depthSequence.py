# OPENCV_IO_ENABLE_OPENEXR=1

import cv2
import numpy as np
import glob
import os
import math

frequency = 2e7

frame_folders = glob.glob("frame*")

if not os.path.exists("depth"):
    os.makedirs("depth")
if not os.path.exists("ir"):
    os.makedirs("ir")
if not os.path.exists("ate"):
    os.makedirs("ate")

for fr in frame_folders:


    r = []
    for i in range(4):
        img_path = f"{fr}/image_phase{i}.exr"

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        if img.shape[2] < 3:
            raise ValueError(f"Image does not have at least 3 channels: {img_path}")

        # Extract red channel (OpenCV uses BGR order)
        red_channel = img[:, :, 2]
        r.append(red_channel)
        
    # intensity = (r[0] + r[1] + r[2] + r[3]) / 4.0

    amplitude = (r[0] - r[2]) ** 2.0 + (r[3] - r[1]) **2.0
    amplitude = np.sqrt(amplitude) * np.pi / 2.0

    phase = np.arctan2(r[3] - r[1], r[0] - r[2])

    depth = np.where(phase < 0, phase + 2 * np.pi, phase)
    depth = (depth / (2 * np.pi)) * 0.5 * (3e8 / frequency)

    depth = np.clip(depth *5000, 0, 65535).astype(np.uint16)

    phase = cv2.normalize(phase, None, 0, 1, cv2.NORM_MINMAX)
    phase = np.clip(phase * 65535, 0, 65535).astype(np.uint16)

    frame_i = int(fr[5:]) # numero de frame, tal como estaba procesado el nombre ("frameX")

    cv2.imwrite(f"ir/{frame_i+1.0:07.1f}.png", amplitude)
    cv2.imwrite(f"depth/{frame_i+1.0:07.1f}.png", depth)

    print(f"Processed {fr}")
