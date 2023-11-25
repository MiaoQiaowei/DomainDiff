data_dir=/data/miaoqiaowei/data/imagenet-r-100
# data_dir=/data/miaoqiaowei/data/imagenet-v2/
# data_dir=/data/yuanjunkun/yjk/dataset/imagenet-r
# output_dir=/home/miaoqiaowei/trex/save/100-100-stable-short_name
output_dir=/home/miaoqiaowei/trex/save/100-100-last_chance
# output_dir=/home/miaoqiaowei/trex/save/100-100_best
# output_dir=/home/miaoqiaowei/trex/save/100-100-stable0.5
# output_dir=/home/miaoqiaowei/trex/save/100-100-intra0.5

# output_dir=/home/miaoqiaowei/trex/save/100-100-intra-extra-ln

export CUDA_VISIBLE_DEVICES=3  # change accordingly the <nproc_per_node> argument below
# RANDOM=10816
RANDOM=22
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 eval.py  \


python3 eval.py  \
    --output_dir=${output_dir} \
    --data_dir=${data_dir} \
    --seed=${RANDOM} \
    --pr_hidden_layers=1 \
    --mc_global_scale 0.40 1.00 \
    --mc_local_scale 0.05 0.40 \
    --n_classes=100 \
    --test \
    --memory_size=0 \

# python3 eval.py  \
#     --output_dir=${output_dir} \
#     --data_dir=${data_dir} \
#     --seed=${RANDOM} \
#     --pr_hidden_layers=3 \
#     --mc_global_scale 0.25 1.00 \
#     --mc_local_scale 0.05 0.25 \
#     --n_classes=100 \
#     --test\
#     --memory_size=0 \
