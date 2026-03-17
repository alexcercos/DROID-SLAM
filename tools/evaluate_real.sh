#!/bin/bash


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
    python evaluation_scripts/test_eth3d.py --datapath=datasets/RealCaptures/loop_path_opposite --weights=checkpoints/$m --disable_vis --testmode=ir --depthmode=depth0.10
    python evaluation_scripts/extract_longitude.py --datapath=datasets/RealCaptures/loop_path_opposite/evaluations/ir_depth0.10.txt

    python evaluation_scripts/test_eth3d.py --datapath=datasets/RealCaptures/loop_path_opposite --weights=checkpoints/$m --disable_vis --testmode=rawir --depthmode=depth --no_use_depth
    python evaluation_scripts/extract_longitude.py --datapath=datasets/RealCaptures/loop_path_opposite/evaluations/rawir_depth_nd.txt
done

