#!/bin/bash


ETH_PATH=datasets/ETH3D-SLAM/training

# all "non-dark" training scenes
evalset=(
    cables_1
    cables_2
    cables_3
#    camera_shake_1
#    camera_shake_2
#    camera_shake_3
    ceiling_1
#    ceiling_2
#    desk_3
#    desk_changing_1
    einstein_1
    einstein_2
    # einstein_dark
#    einstein_flashlight
    einstein_global_light_changes_1
    einstein_global_light_changes_2
#    einstein_global_light_changes_3
    kidnap_1
    # kidnap_dark
    large_loop_1
    mannequin_1
    mannequin_3
    mannequin_4
#    mannequin_5
    mannequin_7
    mannequin_face_1
    mannequin_face_2
#    mannequin_face_3
#    mannequin_head
#    motion_1
    planar_2
    planar_3
    plant_1
    plant_2
    plant_3
    plant_4
    plant_5
    # plant_dark
    plant_scene_1
    plant_scene_2
    plant_scene_3
#    reflective_1
#    repetitive
    sfm_bench
    sfm_garden
    sfm_house_loop
    sfm_lab_room_1
    sfm_lab_room_2
    sofa_1
    sofa_2
    sofa_3
#    sofa_4
    # sofa_dark_1
    # sofa_dark_2
    # sofa_dark_3
    sofa_shake
    table_3
    table_4
    table_7
    vicon_light_1
    vicon_light_2
)

TUM_PATH=datasets/TUM-RGBD

evalset_tum=(
    rgbd_dataset_freiburg1_360
    rgbd_dataset_freiburg1_desk
    rgbd_dataset_freiburg1_desk2
    rgbd_dataset_freiburg1_floor
    rgbd_dataset_freiburg1_plant
    rgbd_dataset_freiburg1_room
#    rgbd_dataset_freiburg1_rpy
    rgbd_dataset_freiburg1_teddy
    rgbd_dataset_freiburg1_xyz
)

GEN_PATH=datasets/Generated

evalset_gen=(
    kitchen_small
    medieval_seaport_1
    medieval_seaport_1v2
    medieval_seaport_loop
    kitchen_table
    country_kitchen
    medieval_seaport_3
    medieval_seaport_internal
    slykdrako_room
    # bathroom
)

# for seq in ${evalset[@]}; do
#     python evaluation_scripts/extract_longitude.py --datapath=$ETH_PATH/$seq
# done


# for seq in ${evalset_tum[@]}; do
#     python evaluation_scripts/extract_longitude.py --datapath=$TUM_PATH/$seq
# done

for seq in ${evalset_gen[@]}; do
    python evaluation_scripts/extract_longitude.py --datapath=$GEN_PATH/$seq
done