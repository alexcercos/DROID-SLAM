CUDA_VISIBLE_DEVICES=0,1,2 python train2.py --name=tartanfull --datapath=/scratch.local2/cercos/train_dataset/ --gpus=3 --usecache --lr=0.00025 --steps=250000
