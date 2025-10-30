#!/bin/bash


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
    python evaluation_scripts/test_eth3d.py --datapath=datasets/RealCaptures/loop_path_opposite --weights=checkpoints/$m --disable_vis --testmode=ir --depthmode=depth0.10
    python evaluation_scripts/extract_longitude.py --datapath=datasets/RealCaptures/loop_path_opposite/evaluations/ir_depth0.10.txt

    python evaluation_scripts/test_eth3d.py --datapath=datasets/RealCaptures/loop_path_opposite --weights=checkpoints/$m --disable_vis --testmode=rawir --depthmode=depth --no_use_depth
    python evaluation_scripts/extract_longitude.py --datapath=datasets/RealCaptures/loop_path_opposite/evaluations/rawir_depth_nd.txt
done

