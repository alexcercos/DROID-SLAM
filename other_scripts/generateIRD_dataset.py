import os
from PIL import Image
import argparse

def create_rgbd_image(rgb_folder, depth_folder, output_folder):
    """
    Creates a new image where:
    - R = Red from RGB image
    - G = Green from RGB image
    - B = Depth image (grayscale)

    Saves output images with the same filenames in the output folder.
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
            depth_image = Image.open(depth_path).convert("L")  # Ensure it's single channel (grayscale)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        # Resize depth to match RGB size if needed
        if rgb_image.size != depth_image.size:
            depth_image = depth_image.resize(rgb_image.size)

        # Split RGB channels
        r, g, _ = rgb_image.split()

        # Create new image from R, G, and depth as B
        merged = Image.merge("RGB", (r, r, depth_image))

        # Save the output
        merged.save(os.path.join(output_folder, filename))

    print(f"RGB-D images saved in: {output_folder}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--output")
    args = parser.parse_args()

    image_folder = os.path.join(args.datapath, 'rgb')
    depth_folder = os.path.join(args.datapath, 'depth')
    
    create_rgbd_image(image_folder, depth_folder, args.output)
