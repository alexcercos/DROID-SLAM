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
    "bothcorr_010000.pth"
    "bothcorr_020000.pth"
    "bothcorr_030000.pth"
    "bothcorr_040000.pth"
    "bothcorr_050000.pth"
    "bothcorr_060000.pth"
    "bothcorr_070000.pth"
    "bothcorr_080000.pth"
    "bothcorr_090000.pth"
    "bothcorr_100000.pth"
    "bothcorr_110000.pth"
    "bothcorr_120000.pth"
    "bothcorr_130000.pth"
    "bothcorr_140000.pth"
    "bothcorr_150000.pth"
    "bothcorr_160000.pth"
    "bothcorr_170000.pth"
    "bothcorr_180000.pth"
    "bothcorr_190000.pth"
    "bothcorr_200000.pth"
    "bothcorr_210000.pth"
    "bothcorr_220000.pth"
    "bothcorr_230000.pth"
    "bothcorr_240000.pth"
    "bothcorr_250000.pth"
)

for m in ${models[@]}; do
    for seq in ${evalset[@]}; do
        python evaluation_scripts/test_eth3d.py --datapath=$GEN_PATH/$seq --weights=checkpoints/$m --disable_vis --testmode=ir --depthmode=zdepth
    done
done
    # for seq in ${evalset[@]}; do
    #     python evaluation_scripts/test_eth3d.py --datapath=$GEN_PATH/$seq --weights=checkpoints/$m --disable_vis --testmode=ir --no_use_depth
    # done
for m in ${models[@]}; do
    for seq in ${evalset[@]}; do
        python evaluation_scripts/test_eth3d.py --datapath=$GEN_PATH/$seq --weights=checkpoints/$m --disable_vis --testmode=ir99-0.05 --depthmode=zdepth
    done
done

#python evaluation_scripts/test_eth3d.py --datapath=datasets/ --weights=checkpoints/$m --disable_vis --testmode=ir
# for seq in ${evalset[@]}; do
#     for th in  ${thresholds[@]}; do
#         python evaluation_scripts/test_threshold.py --datapath=$GEN_PATH/$seq --weights=droid.pth --disable_vis --testmode=ir --depthmode=zdepth --threshold=$th
#     done
# done