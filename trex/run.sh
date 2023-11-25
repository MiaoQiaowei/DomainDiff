# data_dir=/data/yuanjunkun/yjk/dataset/imagenet-100
# data_dir=/data/miaoqiaowei/data/imagenet-100-intra
# output_dir=/home/miaoqiaowei/trex/save/100-100-intra-extra
# export CUDA_VISIBLE_DEVICES=0,1,2,3 # change accordingly the <nproc_per_node> argument below



# data_dir=/data/yuanjunkun/yjk/dataset/imagenet-100
data_dir=/data/miaoqiaowei/data/imagenet-100-all/last_chance
output_dir=/home/miaoqiaowei/trex/save/100-100-last_chance
export CUDA_VISIBLE_DEVICES=0,1,2,3 # change accordingly the <nproc_per_node> argument below


python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 main.py  \
    --output_dir=${output_dir} \
    --data_dir=${data_dir} \
    --seed=${RANDOM} \
    --pr_hidden_layers=1 \
    --mc_global_scale 0.40 1.00 \
    --mc_local_scale 0.05 0.40 \
    --memory_size=0 \
    --n_classes=100 \
    --saveckpt_freq 10\