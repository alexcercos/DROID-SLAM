# OPENCV_IO_ENABLE_OPENEXR=1

import cv2
import numpy as np
import glob
import os
import math
import matplotlib.pyplot as plt

#Display line for each quantile, to see jumps in image histograms 
# (if smooth, could be a valid exposure correction)
def plot_quantiles(images, quantiles):
    
    quantile_data = []
    for _ in quantiles:
        quantile_data.append([])

    for img in images:

        for i,q in enumerate(quantiles):
            quantile_data[i].append(np.percentile(img, q))
    
    for i, data in enumerate(quantile_data):
        plt.plot(data, label=str(quantiles[i]))

    plt.legend()
    #plt.yscale('log')
    plt.show()

def exposure_correction(quantile, pct_margin, img_type, images, names):

    if not os.path.exists(f"{img_type}{quantile}-{pct_margin}"):
        os.makedirs(f"{img_type}{quantile}-{pct_margin}")

    factor = -1.0

    for name, img in zip(names, images):

        q_value = np.percentile(img, quantile)
        max_value = np.max(img)
        
        if factor<0: #first frame
            #set factor at margin or maximum
            factor = min(q_value*(1+pct_margin), max_value)

        else: #other frames

            if q_value < factor * (1-pct_margin):
                factor = factor * (1-pct_margin)
            elif q_value > factor * (1+pct_margin):
                factor = factor * (1+pct_margin)
            else:
                factor = q_value


        frame_i = int(name[5:]) # numero de frame, tal como estaba procesado el nombre ("frameX")

        cv2.imwrite(f"{img_type}{quantile}-{pct_margin}/{frame_i+1.0:07.1f}.png", img / factor * 255.0)

        print(f"Processed {name} factor={factor}")

if __name__ == '__main__':

    ref_quantile = 99
    pct_margin = 0.05 #maximum change allowed in following frames

    frame_folders = sorted(glob.glob("frame*"), key=lambda name: int(name[5:]))


    rgb_images = []
    ir_images = []

    for fr in frame_folders:
        r = []
        rgb = []
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
            rgb.append(img)

        amplitude = (r[0] - r[2]) ** 2.0 + (r[3] - r[1]) **2.0
        amplitude = np.sqrt(amplitude) * np.pi / 2.0

        ir_images.append(amplitude)
        rgb_images.append((rgb[0] + rgb[1] + rgb[2] + rgb[3]) / 4.0)


    exposure_correction(ref_quantile, pct_margin, "ir", ir_images, frame_folders)
    exposure_correction(ref_quantile, pct_margin, "rgb", rgb_images, frame_folders)
    # plot_quantiles(images, [90,100])
