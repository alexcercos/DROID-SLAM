#OPENCV_IO_ENABLE_OPENEXR=1

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

    # cv2.imwrite(f"frame27_int.png", intensity)
    cv2.imwrite(f"ir/{frame_i+1.0:07.1f}.png", amplitude)
    # cv2.imwrite(f"frame27_phase.png", phase)
    cv2.imwrite(f"depth/{frame_i+1.0:07.1f}.png", depth)

    # amplitude_norm = cv2.normalize(amplitude, None, 0, 1, cv2.NORM_MINMAX)
    # amplitude_norm = np.clip(amplitude_norm * 65535, 0, 65535).astype(np.uint16)
    # cv2.imwrite(f"frame27_ampnorm.png", amplitude_norm)

    print(f"Processed {fr}")

gt = []
with open("ground_truth.txt","r") as f:
    gt = f.readlines()

with open("groud_truth_processed.txt","w") as f:

    f.write("# timestamp tx ty tz qx qy qz qw\n")
    
    for fi,frame in enumerate(gt):
        eyeX,eyeY,eyeZ,targetX,targetY,targetZ,uX,uY,uZ = [float(x) for x in frame.split()]

        # Compute quaternion

        #forward
        fX = targetX - eyeX
        fY = targetY - eyeY
        fZ = targetZ - eyeZ

        #right
        rightVec = np.cross([fX,fY,fZ],[uX,uY,uZ])
        rX,rY,rZ =rightVec / np.linalg.norm(rightVec)

        #Should be normalized and orthonormal (1,1,1,0,0,0)
        # print(np.linalg.norm([fX,fY,fZ]),np.linalg.norm([rX,rY,rZ]),np.linalg.norm([uX,uY,uZ]), \
        #       np.dot([fX,fY,fZ],[uX,uY,uZ]),
        #       np.dot([rX,rY,rZ],[uX,uY,uZ]),
        #       np.dot([fX,fY,fZ],[rX,rY,rZ]))

        qx,qy,qz,qw = 0,0,0,0
        trace = rX + uY + fZ

        if trace > 0.0:
            s = 2.0 * math.sqrt(trace + 1.0)
            qw = 0.25 * s
            qx = (uZ - fY) / s
            qy = (fX - rZ) / s
            qz = (rY - uX) / s
        else:
            if (rX > uY) and (rX > fZ):
                s = 2.0 * math.sqrt(1.0 + rX - uY - fZ)
                qw = (uZ - fY) / s
                qx = 0.25 * s
                qy = (uX + rY) / s
                qz = (fX + rZ) / s
            elif uY > fZ:
                s = 2.0 * math.sqrt(1.0 + uY - rX - fZ)
                qw = (fX - rZ) / s
                qx = (uX + rY) / s
                qy = 0.25 * s
                qz = (fY + uZ) / s
            else:
                s = 2.0 * math.sqrt(1.0 + fZ - rX - uY)
                qw = (rY - uX) / s
                qx = (fX + rZ) / s
                qy = (fY + uZ) / s
                qz = 0.25 * s
        
        f.write(f"{fi+1.0:07.1f} {eyeX} {eyeY} {eyeZ} {qx} {qy} {qz} {qw}\n")

    print("Processed ground truth")