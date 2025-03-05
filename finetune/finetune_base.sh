# finetuning script for bert/roberta on GLUE tasks

# =================================================================================================
# Cache directory for storing the model
export CACHE_DIR="/data/models"
export DATA_CACHE_DIR="/data/datasets"

# Output_dir of the finetuned model
export prefix="/home/thw20/projects/AICrossSim"
export OUTPUT_DIR=${prefix}"/ckpt/roberta/finetune/dontcare"

# Model name for the pre-trained model
export MODEL_NAME="JeremiahZ/roberta-base-mrpc"
export MODEL_CONFIG_PATH=/home/thw20/projects/AICrossSim/configs/roberta_base_relu.json
export TASK_LIST="cola mrpc stsb rte sst2 qnli qqp mnli"
export BATCH_SIZE=16
export LR=2e-5

# Set WANDB_DISABLED to true to disable wandb logging
export WANDB_DISABLED=False
# =================================================================================================

function finetune(){
    TASK_NAME=$1
    export RUN_NAME="ft_"$TASK_NAME

    python -u -m torch.distributed.launch --master_port=14400 --nproc_per_node=1 --nnodes=1 --node_rank=0 --use_env \
        finetune/run_glue.py \
        --seed 42 \
        --model_name_or_path $MODEL_NAME \
        --config_name $MODEL_CONFIG_PATH \
        --task_name $TASK_NAME \
        --save_strategy no \
        --evaluation_strategy epoch \
        --cache_dir $CACHE_DIR \
        --data_cache_dir $DATA_CACHE_DIR \
        --do_train \
        --do_eval \
        --fp16 \
        --ddp_timeout 180000 \
        --max_seq_length 512 \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --learning_rate $LR \
        --num_train_epochs 10 \
        --output_dir $OUTPUT_DIR \
        --report_to wandb \
        --run_name $RUN_NAME \
        --wandb-tags "$TASK_NAME relu"  \
        --overwrite_output_dir
        # Uncomment the following lines to enable wandb logging
        # --report_to wandb \
        # --run_name $RUN_NAME \
        # --wandb-tags "$TASK_NAME"
}



finetune "mrpc"

# for task in $TASK_LIST
# do
#     finetune $task
# done

