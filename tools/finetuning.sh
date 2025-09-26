
# ./tools/finetuning.sh | tee -a TRAIN.log

#1: Unfreeze 1 layer of fnet and cnet, and all of update
CUDA_VISIBLE_DEVICES=1,2 python train.py --name=test1 --ckpt=droid.pth --datapath=datasets/Generated --unfreeze_cnet=1 --unfreeze_fnet=1 --unfreeze_update=agg,gru,weight,delta,corr,flow

#2: Unfreeze 2 layers of fnet and cnet, and all of update
CUDA_VISIBLE_DEVICES=1,2 python train.py --name=test2 --ckpt=droid.pth --datapath=datasets/Generated --unfreeze_cnet=2 --unfreeze_fnet=2 --unfreeze_update=agg,gru,weight,delta,corr,flow

#3: Unfreeze 2 layers of fnet and cnet, no update
CUDA_VISIBLE_DEVICES=1,2 python train.py --name=test3 --ckpt=droid.pth --datapath=datasets/Generated --unfreeze_cnet=2 --unfreeze_fnet=2

#4: Unfreeze update, no fnet or cnet
CUDA_VISIBLE_DEVICES=1,2 python train.py --name=test4 --ckpt=droid.pth --datapath=datasets/Generated --unfreeze_update=agg,gru,weight,delta,corr,flow

#5: Unfreeze 2 layer of fnet and cnet, weight and delta update
CUDA_VISIBLE_DEVICES=1,2 python train.py --name=test5 --ckpt=droid.pth --datapath=datasets/Generated --unfreeze_cnet=2 --unfreeze_fnet=2 --unfreeze_update=weight,delta

#6: Unfreeze 3 layers of fnet and cnet, and all of update
CUDA_VISIBLE_DEVICES=1,2 python train.py --name=test6 --ckpt=droid.pth --datapath=datasets/Generated --unfreeze_cnet=3 --unfreeze_fnet=3 --unfreeze_update=agg,gru,weight,delta,corr,flow
