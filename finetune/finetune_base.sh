# finetuning script for bert/roberta on GLUE tasks

export prefix="/data/models/thw20/aixsim"
export CACHE_DIR="/data/models"
export DATA_CACHE_DIR="/data/datasets"
export OUTPUT_DIR=${prefix}"/ckpt/roberta/finetune/dontcare"
export MODEL_NAME="FacebookAI/roberta-base"
export TASK_LIST="cola mrpc stsb rte sst2 qnli qqp mnli"
# Set WANDB_DISABLED to true to disable wandb logging
export WANDB_DISABLED=true

function finetune(){
    TASK_NAME=$1
    export RUN_NAME="ft_"$TASK_NAME

    python -u -m torch.distributed.launch --master_port=14400 --nproc_per_node=1 --nnodes=1 --node_rank=0 --use_env \
        run_glue.py \
        --model_name_or_path $MODEL_NAME \
        --task_name $TASK_NAME \
        --save_strategy no \
        --cache_dir $CACHE_DIR \
        --data_cache_dir $DATA_CACHE_DIR \
        --do_train \
        --do_eval \
        --fp16 \
        --ddp_timeout 180000 \
        --max_seq_length 512 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --learning_rate 2e-5 \
        --num_train_epochs 10 \
        --output_dir $OUTPUT_DIR \
        --overwrite_output_dir
        # Uncomment the following lines to enable wandb logging
        # --report_to wandb \
        # --run_name $RUN_NAME \
        # --wandb-tags "$TASK_NAME"
}


for task in $TASK_LIST
do
    finetune $task
done

