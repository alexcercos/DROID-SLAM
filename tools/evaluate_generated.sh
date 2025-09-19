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

for seq in ${evalset[@]}; do
    for th in  ${thresholds[@]}; do
        python evaluation_scripts/test_threshold.py --datapath=$GEN_PATH/$seq --weights=droid.pth --disable_vis --testmode=ir --depthmode=zdepth --threshold=$th
    done
done