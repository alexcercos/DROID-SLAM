import glob
import argparse
import os

def generate_coordinated_depth(rgb_folder, org_folder, depth_folder):

    if not os.path.exists(depth_folder):
        os.makedirs(depth_folder)

    rgb_files = sorted(os.listdir(rgb_folder))
    depth_files = sorted(os.listdir(org_folder))

    current_depth = 0 # depth index

    all_diffs = []

    max_frame_diff = 0
    timestamp = 0

    for i,filename in enumerate(rgb_files):
        if not filename.lower().endswith('.png'):
            continue
        
        prev_ts = timestamp
        timestamp = float(filename.replace(".png",""))

        if i>0:
            max_frame_diff = max(max_frame_diff, timestamp-prev_ts)
        
        current_diff = abs(timestamp - float(depth_files[current_depth].replace(".png","")))
        diff = timestamp
        if current_depth < len(depth_files)-1:
            diff = abs(timestamp - float(depth_files[current_depth+1].replace(".png","")))

        while diff < current_diff:
            current_depth+=1
            current_diff = diff
            if current_depth < len(depth_files)-1:
                diff = abs(timestamp - float(depth_files[current_depth+1].replace(".png","")))

        #Copy file
        os.system(f"cp {org_folder}/{depth_files[current_depth]} {depth_folder}/{filename}")

        all_diffs.append((current_diff, f"{i}: {filename} / {depth_files[current_depth]}"))

    #Print sorted diffs
    all_diffs.sort(key=lambda x: x[0])
    for t,f in all_diffs:
        print(t,f)
    print(f"Max frame diff = {max_frame_diff}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    
    args = parser.parse_args()

    image_folder = os.path.join(args.datapath, 'rgb')
    depth_folder = os.path.join(args.datapath, 'depth_org')
    output_folder = os.path.join(args.datapath, "depth")
    
    generate_coordinated_depth(image_folder, depth_folder, output_folder)
