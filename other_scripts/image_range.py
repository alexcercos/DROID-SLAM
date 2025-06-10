from PIL import Image
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# Gather all image paths
image_list = sorted(glob.glob(os.path.join("datasets", "ETH3D-SLAM", "**", "depth", '*.png'), recursive=True))

total = len(image_list)
abs_min = np.iinfo(np.uint32).max
abs_max = 0

# We accumulate histogram in a fixed-size array, assuming 16-bit max (65536)
hist_size = 2**16
global_hist = np.zeros(hist_size, dtype=np.float64)

for i, image_path in enumerate(image_list):
    img = Image.open(image_path)
    img_array = np.array(img)

    non_zero = img_array[img_array > 0]
    if non_zero.size > 0:
        min_val = non_zero.min()
        max_val = img_array.max()
        abs_min = min(abs_min, min_val)
        abs_max = max(abs_max, max_val)

        # Update histogram only for non-zero pixels
        hist = np.bincount(non_zero.flatten(), minlength=hist_size)
        # Update histogram only for non-zero pixels
        hist = np.bincount(non_zero.flatten().astype(np.uint32), minlength=hist_size)
        global_hist[:len(hist)] += hist

    print(f"{i+1}/{total}: {abs_min}/{abs_max} ({abs_min/5000.0:.3f} / {abs_max/5000.0:.3f})    ", end="\r")

print("\nFinished processing.")

# Plot histogram
x = np.arange(hist_size)
plt.figure(figsize=(12, 6))
plt.plot(x[abs_min:abs_max+1], global_hist[abs_min:abs_max+1])
plt.title("Depth Value Distribution (Non-zero Pixels)")
plt.xlabel("Depth Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()