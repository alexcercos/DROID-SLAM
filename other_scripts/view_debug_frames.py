import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

def natural_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', s)
    ]

# ==========================
# Configuration
# ==========================

root_dir = "."
window_name = "Corr-Depth corrs (K=10,100) / Disps, disps sens, images"

IMAGE_LIST = [
    ["corr_i_mean.png", "depth_corr_i_mean.png", "depth_corr_i_mean_k100.png"], 
    ["disp_i.png", "disp_j.png", "image_i.png"],
    ["disp_sens_i.png", "disp_sens_j.png", "image_j.png"]
]

# ==========================
# Index filesystem once
# ==========================

root = Path(root_dir)

current_index = None
current_iterator = 0
looping = True
#organize with values -> iteration - upd it. - fr I - fr J
#switch i-j

index_map = {}
iterations = []

for iter_dir in sorted((p for p in root.iterdir() if p.is_dir()),key=lambda p: natural_key(p.name)):

    fr, upd_l, upd_i = iter_dir.name.split('-')
    if len(iterations)==0 or fr != iterations[-1]:
        iterations.append(fr)

    for frame_dir in iter_dir.iterdir():
        if frame_dir.is_dir():

            ii,jj = frame_dir.name.split('-')

            if fr not in index_map:
                index_map[fr] = {}
            if (upd_l,upd_i) not in index_map[fr]:
                index_map[fr][f"{upd_l}-{upd_i}"] = []

            index_map[fr][f"{upd_l}-{upd_i}"].append((int(ii),int(jj)))

            if current_index is None:
                current_index = [0, int(upd_l), int(upd_i), int(ii), int(jj)]

if not index_map:
    raise RuntimeError("No frames found")

frame_list = sorted(index_map.keys())

# ==========================
# Image loading
# ==========================

def load_images():

    fr_i, upd_l, upd_i, ii, jj = current_index

    fr = iterations[fr_i]

    frame_dir = root / f"{fr}-{upd_l}-{upd_i}" / f"{ii}-{jj}"

    combined = []

    for row in IMAGE_LIST:
        row_list = []
        for img in row:
            loaded_img = cv2.imread(str(frame_dir / img), cv2.IMREAD_COLOR)
            row_list.append(loaded_img)
        combined.append(np.hstack(row_list))


    return np.vstack(combined)

def iterate(increase):

    global current_index
    global index_map
    
    if current_iterator == 0:
        current_index[0] = (current_index[0] + increase + len(iterations)) % len(iterations)
    else:
        current_index[current_iterator] += increase

    fr_i, upd_l, upd_i, ii, jj = current_index

    fr = iterations[fr_i]

    upd_dir = root / f"{fr}-{upd_l}-{upd_i}"

    if not upd_dir.exists():
        if upd_l == 2 and current_iterator != 1:
            upd_l = 1
            upd_i = 3
        elif upd_l >= 2:
            upd_l = 1
            upd_i = 0
        elif upd_i > 3:
            upd_i = 0
        elif upd_i < 0:
            upd_i = 3
        upd_dir = root / f"{fr}-{upd_l}-{upd_i}"

    if not upd_dir.exists():
        upd_i = 0
        upd_l = 1
        upd_dir = root / f"{fr}-{upd_l}-{upd_i}"

    assert upd_dir.exists(), upd_dir.name

    frame_dir = upd_dir / f"{ii}-{jj}"
    if not frame_dir.exists():
        
        frame_list = index_map[fr][f"{upd_l}-{upd_i}"]

        best_match = (ii,jj)
        for fi,fj in frame_list:
            curr_d = abs(fi-ii) + abs(fj-jj)
            best_match = (fi,fj)

        ii,jj = best_match

    frame_dir = upd_dir / f"{ii}-{jj}"
    assert frame_dir.exists(), frame_dir.name

    current_index = [fr_i, upd_l, upd_i, ii, jj]


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
        current_iterator = (current_iterator + 1) % len(current_index)

    elif key == ord('a') or key == 81:  # left
        current_iterator = (current_iterator - 1 + len(current_index)) % len(current_index)

    elif key == ord('w') or key == 82:  # next iteration
        iterate(1)

    elif key == ord('s') or key == 84:  # previous iteration
        iterate(-1)

    # elif key == ord('l'): #Toggle looping
    #     looping = not looping
    
    elif key == ord('x'): #switch i-j
        i,j = current_index[-2:]
        current_index[-2] = j
        current_index[-1] = i

    elif key == ord('q') or key == 27:
        break

    # ======================
    # Status line
    # ======================

    print_line = iterations[current_index[0]]+" / "
    for i,n in enumerate(current_index):
        if current_iterator==i:
            print_line+=f"[{n}] "
        else:
            print_line+=f"{n} "

    print(print_line+"      ", end="\r")

cv2.destroyAllWindows()
