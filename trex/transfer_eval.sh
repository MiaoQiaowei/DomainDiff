# ckpt_file=/home/miaoqiaowei/trex/save/100-100-stable-good/checkpoint.pth
ckpt_file=/home/miaoqiaowei/trex/save/100-100-intra0.5/checkpoint_best.pth
# if you downloaded the full checkpoint, set ckpt_key="model"
# otherwise (if you downloaded ResNet50 checkpoint), set it as ckpt_key="none"
# 'in1k', 'cog_l1', 'cog_l2', 'cog_l3', 'cog_l4', 'cog_l5', 'aircraft', 'cars196', 'dtd', 'eurosat', 'flowers', 'pets', 'food101', 'sun397', 'inat2018', 'inat2019')
ckpt_key="model"
dataset="sun397"
dataset_dir=/data/miaoqiaowei/data/$dataset
# output_dir=/home/miaoqiaowei/trex/features/$dataset
features_dir=/home/miaoqiaowei/trex/features/$dataset
output_dir=/home/miaoqiaowei/trex/save/$dataset
export CUDA_VISIBLE_DEVICES=0,1,2,3  # set this to "" for CPU-only feature extraction

python feature_extraction.py \
    --ckpt_file=${ckpt_file} \
    --ckpt_key=${ckpt_key} \
    --dataset=${dataset} \
    --dataset_dir=${dataset_dir} \
    --output_dir=${features_dir} \

# L2 normalization follows the convention in the ImageNet-CoG benchmark.
python classifier_training.py \
    --features_dir=${features_dir} \
    --features_norm="l2" \
    --clf_type="logreg_torch" \
    --output_dir=${output_dir} \
    --seed=${RANDOM}