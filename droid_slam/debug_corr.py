import torch
import numpy as np
import os

import matplotlib.pyplot as plt

folder = "CURRENT_DEBUG"

disp_div = 1

def save_frame_info(frame_name, ii, jj, corr, depth_corr, images, disps, disps_sens, poses):

    ii = ii.cpu().numpy()
    jj = jj.cpu().numpy()
    corr = corr.cpu().numpy()
    depth_corr = depth_corr.cpu().numpy()
    images = images.cpu().numpy()
    disps = disps.cpu().numpy()
    disps_sens = disps_sens.cpu().numpy()
    poses = poses.cpu().numpy()
    
    for n,(ix,jx) in enumerate(zip(ii,jj)):
        subfolder = f"{folder}/{frame_name}/{ix}-{jx}"
        #Create subfolder for each pair i-j
        os.makedirs(subfolder, exist_ok=True)

        corr_i        = corr[:, n, :, :, :]         # (1, 196, 55, 55)
        depth_corr_i  = depth_corr[:, n, :, :, :]   # (1, 196, 55, 55)
        disp_i        = disps[ix]                   # (55, 55)
        disp_sens_i   = disps_sens[ix]              # (55, 55)
        disp_j        = disps[jx]                   # (55, 55)
        disp_sens_j   = disps_sens[jx]              # (55, 55)


        # print(f"min: {disp_j.min():.6f}, max: {disp_j.max():.6f}, mean: {disp_j.mean():.6f}")

        # Save raws just in case
        np.save(f"{subfolder}/corr_raw.npy", corr[:, n, :, :, :])
        np.save(f"{subfolder}/depth_corr_raw.npy", depth_corr[:, n, :, :, :])
        np.save(f"{subfolder}/poses.npy", poses[[ix,jx]])
        np.save(f"{subfolder}/disps_raw.npy", disps[[ix,jx]])
        np.save(f"{subfolder}/disps_sens_raw.npy", disps_sens[[ix,jx]])

        # Save images
        plt.imsave(f"{subfolder}/image_i.png", np.transpose(images[ix], (1, 2, 0)).astype(np.uint8)) # (3, 440, 440)
        plt.imsave(f"{subfolder}/image_j.png", np.transpose(images[jx], (1, 2, 0)).astype(np.uint8)) # (3, 440, 440)

        # Correlations
        for c_corr,name in [(corr_i,"corr_i"),(depth_corr_i,"depth_corr_i")]:
            corr_avg = c_corr.mean(axis=1).squeeze(0)
            corr_clamped = np.clip(corr_avg, 0.0, 1.0)
            corr_color = plt.cm.jet(corr_clamped)[..., :3]
            corr_up = np.repeat(np.repeat(corr_color, 8, axis=0),8, axis=1)
            plt.imsave(f"{subfolder}/{name}_mean.png", corr_up)


        for disp,name in [(disp_i,"disp_i"),(disp_j,"disp_j"),(disp_sens_i,"disp_sens_i"),(disp_sens_j,"disp_sens_j")]:
            disp_clamped = np.clip(disp / disp_div, 0.0, 1.0)
            disp_color = plt.cm.jet(disp_clamped)[..., :3]
            disp_up = np.repeat(np.repeat(disp_color, 8, axis=0),8, axis=1)
            plt.imsave(f"{subfolder}/{name}.png", disp_up)
