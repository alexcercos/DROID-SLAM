import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# ==========================
# Configuration
# ==========================

root_dir = "."
window_name = "Image (Left) + Disparity (Right)"

IMAGE_LIST = [
    ["corr_i_mean.png", "depth_corr_i_mean.png", "depth_corr_i_mean_k100.png"], 
    ["disp_i.png", "disp_j.png", "image_i.png"],
    ["disp_sens_i.png", "disp_sens_j.png", "image_j.png"]
]

# ==========================
# Index filesystem once
# ==========================

root = Path(root_dir)

# index_map[frame] = sorted list of iterations that contain that frame
index_map = defaultdict(list)

for iter_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    for frame_dir in iter_dir.iterdir():
        if frame_dir.is_dir():
            index_map[frame_dir.name].append(iter_dir.name)

# sort + unique
for frame in index_map:
    index_map[frame] = sorted(set(index_map[frame]))

if not index_map:
    raise RuntimeError("No frames found")

frame_list = sorted(index_map.keys())

# ==========================
# Viewer state
# ==========================

frame_idx = 0
iter_idx = 0

def clamp_iter_idx():
    global iter_idx
    iters = index_map[frame_list[frame_idx]]
    iter_idx = max(0, min(iter_idx, len(iters) - 1))

clamp_iter_idx()

# ==========================
# Image loading
# ==========================

def load_images():
    frame = frame_list[frame_idx]
    iteration = index_map[frame][iter_idx]

    frame_dir = root / iteration / frame

    combined = []

    for row in IMAGE_LIST:
        row_list = []
        for img in row:
            loaded_img = cv2.imread(str(frame_dir / img), cv2.IMREAD_COLOR)
            row_list.append(loaded_img)
        combined.append(np.hstack(row_list))


    return np.vstack(combined)

# ==========================
# Main loop
# ==========================

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    combined = load_images()
    cv2.imshow(window_name, combined)

    key = cv2.waitKey(1) & 0xFF

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    # ======================
    # Frame navigation
    # ======================
    if key == ord('d') or key == 83:  # right
        frame_idx = (frame_idx + 1) % len(frame_list)
        clamp_iter_idx()

    elif key == ord('a') or key == 81:  # left
        frame_idx = (frame_idx - 1) % len(frame_list)
        clamp_iter_idx()

    # ======================
    # Iteration navigation
    # ======================
    elif key == ord('e'):  # next iteration
        iters = index_map[frame_list[frame_idx]]
        iter_idx = (iter_idx + 1) % len(iters)

    elif key == ord('w'):  # previous iteration
        iters = index_map[frame_list[frame_idx]]
        iter_idx = (iter_idx - 1) % len(iters)

    # ======================
    # Quit
    # ======================
    elif key == ord('q') or key == 27:
        break

    # ======================
    # Status line
    # ======================
    frame = frame_list[frame_idx]
    iteration = index_map[frame][iter_idx]
    print(
        f"Frame {frame_idx+1}/{len(frame_list)}: {frame} | "
        f"Iter {iter_idx+1}/{len(index_map[frame])}: {iteration} | ",
        end="\r"
    )

cv2.destroyAllWindows()
