#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
domain="imagenet-100"
# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export MODEL_NAME="/home/miaoqiaowei/.cache/huggingface/models--stabilityai--stable-diffusion-2-1-base/snapshots/88bb1a46821197d1ac0cb54d1d09fb6e70b171bc"
export INSTANCE_DIR="/data/miaoqiaowei/data/imagenet-100/finetune"
export OUTPUT_DIR="../output/$domain-fine_photo"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

accelerate launch d2d_train_lora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="photo" \
  --resolution=224 \
  --train_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000