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

for seq in ${evalset[@]}; do
    python evaluation_scripts/test_eth3d.py --datapath=$GEN_PATH/$seq --weights=droid.pth --disable_vis --testmode=depth --depthmode=depth --no_use_depth
done