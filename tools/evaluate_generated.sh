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
    "tartanfull_010000.pth"
    "tartanfull_020000.pth"
    "tartanfull_030000.pth"
    "tartanfull_040000.pth"
    "tartanfull_050000.pth"
    "tartanfull_060000.pth"
    "tartanfull_070000.pth"
    "tartanfull_080000.pth"
    "tartanfull_090000.pth"
    "tartanfull_100000.pth"
    "tartanfull_110000.pth"
    "tartanfull_120000.pth"
    "tartanfull_130000.pth"
    "tartanfull_140000.pth"
    "tartanfull_150000.pth"
    "tartanfull_160000.pth"
    "tartanfull_170000.pth"
    "tartanfull_180000.pth"
    "tartanfull_190000.pth"
    "tartanfull_200000.pth"
)

for m in ${models[@]}; do
    for seq in ${evalset[@]}; do
        python evaluation_scripts/test_eth3d.py --datapath=$GEN_PATH/$seq --weights=checkpoints/$m --disable_vis --testmode=ir --depthmode=zdepth
    done

    for seq in ${evalset[@]}; do
        python evaluation_scripts/test_eth3d.py --datapath=$GEN_PATH/$seq --weights=checkpoints/$m --disable_vis --testmode=ir --no_use_depth
    done

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