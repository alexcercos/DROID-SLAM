import os
from PIL import Image
import numpy as np
import cv2

import argparse

def create_rgbd_image(rgb_folder, depth_folder, output_folder, depth_max_value=10000):
    """
    Creates new RGB images where:
    - R = Red from RGB image
    - G = Red from RGB image
    - B = Depth (normalized from 16-bit to 8-bit)

    Parameters:
    - depth_max_value: max expected depth value (used for normalization)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filenames = sorted(os.listdir(rgb_folder))

    for filename in filenames:
        if not filename.lower().endswith('.png'):
            continue

        rgb_path = os.path.join(rgb_folder, filename)
        depth_path = os.path.join(depth_folder, filename)

        try:
            rgb_image = Image.open(rgb_path).convert("RGB")
            depth_image = Image.open(depth_path)
            depth_np = np.array(depth_image, dtype=np.uint16)

            # Normalize 16-bit depth to 8-bit for B channel
            depth_normalized = np.clip(depth_np / depth_max_value * 255.0, 0, 255).astype(np.uint8)
            depth_img_8bit = Image.fromarray(depth_normalized, mode='L')

            # Use R and G from RGB, B from depth
            r, g, _ = rgb_image.split()
            merged = Image.merge("RGB", (r, r, depth_img_8bit))

            # Save output
            merged.save(os.path.join(output_folder, filename))

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    print(f"RGB-D images saved in: {output_folder}")

def rgb_to_average_grayscale(rgb_folder, output_folder):
    """
    Converts an RGB image to a grayscale image by averaging R, G, B channels.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filenames = sorted(os.listdir(rgb_folder))

    for filename in filenames:
        if not filename.lower().endswith('.png'):
            continue
        rgb_path = os.path.join(rgb_folder, filename)
        # Open and convert to RGB (in case it's not)
        image = Image.open(rgb_path).convert("RGB")
        
        # Convert to NumPy array
        rgb_np = np.array(image, dtype=np.uint8)

        # Average across the color channels (axis=2)
        gray_np = rgb_np.mean(axis=2).astype(np.uint8)

        # Convert back to image
        gray_image = Image.fromarray(gray_np, mode='L')

        # Save
        gray_image.save(os.path.join(output_folder, filename))

    print(f"Grayscale images saved in: {output_folder}")

def fill_black_with_nearest(filename):
    """
    Fill black (0-value) pixels in a 16-bit depth image with the value of the nearest non-zero neighbor.
    
    Parameters:
        depth_img (np.ndarray): 2D NumPy array of dtype np.uint16 representing the depth image.
    
    Returns:
        np.ndarray: Depth image with black pixels filled.
    """
    depth_image = Image.open(filename)
    depth_img = np.array(depth_image, dtype=np.uint16)

    # Create mask of black pixels (0 value)
    mask = depth_img == 0

    # Create a binary image for distance transform (non-zero where depth != 0)
    non_zero_mask = (~mask).astype(np.uint8)

    # Compute distance transform and labels
    dist, labels = cv2.distanceTransformWithLabels(
        1 - non_zero_mask,  # invert: treat zero as foreground
        distanceType=cv2.DIST_L2,
        maskSize=5,
        labelType=cv2.DIST_LABEL_PIXEL
    )

    # Find the coordinates of all non-zero pixels
    coords = np.column_stack(np.where(non_zero_mask > 0))

    # Map label -> coordinates (labels start from 1)
    label_to_coords = {}
    for i, (y, x) in enumerate(coords):
        label = labels[y, x]
        if label not in label_to_coords:
            label_to_coords[label] = (y, x)

    # Create output image
    filled_img = depth_img.copy()

    # Fill each black pixel with value from nearest non-black pixel
    black_coords = np.column_stack(np.where(depth_img == 0))
    for y, x in black_coords:
        nearest_label = labels[y, x]
        if nearest_label == 0:
            continue  # skip if no label found (shouldn't happen unless entire image is 0)
        ny, nx = label_to_coords.get(nearest_label, (y, x))  # fallback to self
        filled_img[y, x] = depth_img[ny, nx]

    return filled_img

def create_fill_depth(depth_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filenames = sorted(os.listdir(depth_folder))

    for filename in filenames:
        if not filename.lower().endswith('.png'):
            continue
        depth_path = os.path.join(depth_folder, filename)

        filled_img = fill_black_with_nearest(depth_path)
        Image.fromarray(filled_img, mode='I;16').save(os.path.join(output_folder, filename))

    print(f"Filled depth images saved in: {output_folder}")

def atenuated_infrarred(bn_folder, fdepth_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filenames = sorted(os.listdir(bn_folder))

    for filename in filenames:
        if not filename.lower().endswith('.png'):
            continue

        bn_path = os.path.join(bn_folder, filename)
        depth_path = os.path.join(fdepth_folder, filename)

        try:
            bn_image = Image.open(bn_path).convert("L")
            bn_img = np.array(bn_image, dtype=np.float32)

            depth_image = Image.open(depth_path)
            depth_np = np.array(depth_image, dtype=np.float32)

            depth_np = np.divide(depth_np, 20000)
            depth_exp = np.power(np.minimum(depth_np, 1), 2)

            atenuated_img = bn_img * (1.0 - depth_exp)

            Image.fromarray(atenuated_img.astype(np.uint8), mode='L').save(os.path.join(output_folder, filename))

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    print(f"RGB-D images saved in: {output_folder}")


#TODO: noisy depth images?

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--output")
    parser.add_argument("--test", action="store_true")
    
    args = parser.parse_args()

    if not args.test:

        image_folder = os.path.join(args.datapath, 'rgb')
        depth_folder = os.path.join(args.datapath, 'depth')
        output_folder = os.path.join(args.datapath, args.output)
        
        create_fill_depth(depth_folder, output_folder)
    
    else:

        image_folder = os.path.join(args.datapath, 'bn')
        depth_folder = os.path.join(args.datapath, 'fdepth')
        output_folder = os.path.join(args.datapath, args.output)
        
        atenuated_infrarred(image_folder, depth_folder, output_folder)
