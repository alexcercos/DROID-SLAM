#!/bin/bash


GEN_PATH=datasets/Generated

evalset=(
    kitchen_small
    medieval_seaport_1
    medieval_seaport_1v2
    medieval_seaport_loop
    kitchen_table
    country_kitchen
    medieval_seaport_3
    medieval_seaport_internal
    slykdrako_room
    bathroom
    country_kitchen_slow
    kitchen_slow
)

thresholds=(
    0
    10
    20
    30
    40
    50
    60
    70
    80
    90
    100
)

models=(
    "test1_010000.pth"
    "test2_010000.pth"
    "test3_010000.pth"
    "test4_010000.pth"
    "test5_010000.pth"
)

for seq in ${evalset[@]}; do
    for m in ${models[@]}; do
        python evaluation_scripts/test_eth3d.py --datapath=$GEN_PATH/$seq --weights=checkpoints/$m --disable_vis --testmode=ir --upsample --reconstruction_path=$seq_ir_$m
    done
done
#python evaluation_scripts/test_eth3d.py --datapath=datasets/ --weights=checkpoints/$m --disable_vis --testmode=ir
# for seq in ${evalset[@]}; do
#     for th in  ${thresholds[@]}; do
#         python evaluation_scripts/test_threshold.py --datapath=$GEN_PATH/$seq --weights=droid.pth --disable_vis --testmode=ir --depthmode=zdepth --threshold=$th
#     done
# done