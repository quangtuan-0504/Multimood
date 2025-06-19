#!/bin/bash

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${1:-4}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=${3:-0}

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=4
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]

# Log Arguments
export WANDB_PROJECT=videollama2qwen2_downstream_sft
RUN_NAME=siglip_tcv35_7b_16f_qlora
DATA_DIR=datasets
OUTP_DIR=work_dirs

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    framework/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 --bits 4 \
    --deepspeed scripts/zero2.json \
    --model_type videollama2_qwen2 \
    --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-AV \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector_type stc_connector_v35 \
    --data_path /workspace/SMES_Therapy_framework/data/mesc/train.jsonl \
    --vid_folder /workspace/SMES_Therapy_framework/data/mesc/video_data/ \
    --mamba_compressor /workspace/SMES_Therapy_framework/mamba_checkpoints \
    --freeze_mamba_compressor true \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --num_frames 8 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/finetune_${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 99 \
    --learning_rate 3e-7 \
    --weight_decay 0.3 \
    --warmup_ratio 0.03 \
    --max_grad_norm 5.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --report_to tensorboard \
    --run_name $RUN_NAME \
