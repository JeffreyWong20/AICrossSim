# finetuning script for bert/roberta on GLUE tasks

# =================================================================================================
# Cache directory for storing the model
export CACHE_DIR="/data/models"
export DATA_CACHE_DIR="/data/datasets"

# Output_dir of the finetuned model
export prefix="/home/thw20/projects/AICrossSim"

# Mpdel config 
# export MODEL_NAME_PREFIX="JeremiahZ/roberta-base-"
export MODEL_NAME_PREFIX="JeffreyWong/roberta-base-relu-"
export TRAINING_DISABLE=true
export USE_RELU=true

# Training parameters
export BATCH_SIZE=16

# single task
export USE_SINGLE_TASK=true
export USE_CKPT=true
export USE_MASE=1
export MODEL_CHPT_PATH="/home/thw20/projects/AICrossSim/ckpt/roberta/finetune/mrpc/lr_1e-5"
export TASK_NAME="mrpc"
export LR=1e-5

# multi-task
export TASK_LIST="cola stsb rte sst2 qqp"
export LR_LIST="1e-5 2e-5 3e-5"

# push to hub
export PUSH_TO_HUB=false
# =================================================================================================

function finetune(){
    TASK_NAME=$1
    export RUN_NAME="ft_"$TASK_NAME"_"$LR

    if [ "$TRAINING_DISABLE" = false ]; then
        python -u -m torch.distributed.launch --master_port=14400 --nproc_per_node=1 --nnodes=1 --node_rank=0 --use_env \
        finetune/run_glue.py \
            --seed 42 \
            --model_name_or_path $MODEL_NAME \
            --config_name $MODEL_CONFIG_PATH \
            --task_name $TASK_NAME \
            --save_strategy best \
            --save_total_limit 2 \
            --eval_strategy epoch \
            --metric_for_best_model $BEST_METRIC \
            --load_best_model_at_end \
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
            --wandb-tags "$TASK_NAME relu $LR spikezip"  \
            --use_mase $USE_MASE
            --overwrite_output_dir
            # Uncomment the following lines to enable wandb logging
            # --report_to wandb \
            # --run_name $RUN_NAME \
            # --wandb-tags "$TASK_NAME"
    elif [ "$PUSH_TO_HUB" = true ]; then
            python -u -m torch.distributed.launch --master_port=20000 --nproc_per_node=1 --nnodes=1 --node_rank=0 --use_env \
                finetune/run_glue_push_to_hub.py \
                --seed 42 \
                --model_name_or_path $MODEL_NAME \
                --config_name $MODEL_CONFIG_PATH \
                --task_name $TASK_NAME \
                --save_strategy no \
                --cache_dir $CACHE_DIR \
                --data_cache_dir $DATA_CACHE_DIR \
                --do_eval \
                --fp16 \
                --ddp_timeout 180000 \
                --max_seq_length 512 \
                --per_device_eval_batch_size 4 \
                --output_dir $OUTPUT_DIR \
                --overwrite_output_dir \
                --use_mase $USE_MASE \
                --push_to_hub
    else
        python -u -m torch.distributed.launch --master_port=14400 --nproc_per_node=1 --nnodes=1 --node_rank=0 --use_env \
            finetune/run_glue.py \
            --seed 42 \
            --model_name_or_path $MODEL_NAME \
            --config_name $MODEL_CONFIG_PATH \
            --task_name $TASK_NAME \
            --save_strategy no \
            --cache_dir $CACHE_DIR \
            --data_cache_dir $DATA_CACHE_DIR \
            --do_eval \
            --fp16 \
            --ddp_timeout 180000 \
            --max_seq_length 512 \
            --per_device_eval_batch_size 4 \
            --output_dir $OUTPUT_DIR \
            --use_mase $USE_MASE \
            --overwrite_output_dir
    fi
}


# Set WANDB_DISABLED to true to disable wandb logging
export WANDB_DISABLED=$TRAINING_DISABLE

if [ "$USE_RELU" = true ]; then
    export MODEL_CONFIG_PATH=/home/thw20/projects/AICrossSim/configs/roberta_base_relu.json
else
    export MODEL_CONFIG_PATH=$MODEL_NAME
fi

if [ "$USE_SINGLE_TASK" = true ]; then
    echo "Single task finetuning"
    if [ "$PUSH_TO_HUB" = true ]; then
        export OUTPUT_DIR=${prefix}"/ckpt/roberta/finetune/"$TASK_NAME"/roberta-base-relu-"$TASK_NAME
    else
        export OUTPUT_DIR=${prefix}"/ckpt/roberta/finetune/"$TASK_NAME"/lr_"$LR
    fi

    if [ "$USE_CKPT" = true ]; then
        export MODEL_NAME=$MODEL_CHPT_PATH
    else
        export MODEL_NAME=$MODEL_NAME_PREFIX$TASK_NAME
    fi
    echo $MODEL_NAME
    echo "Finetuning on "$TASK_NAME" with LR="$LR" and output_dir="$OUTPUT_DIR" and model_name="$MODEL_NAME


    if [ $TASK_NAME = "mnli" ]; then
        export BEST_METRIC="accuracy"
    elif [ $TASK_NAME = "mrpc" ]; then
        export BEST_METRIC="accuracy"
    elif [ $TASK_NAME = "sst2" ]; then
        export BEST_METRIC="accuracy"
    elif [ $TASK_NAME = "cola" ]; then
        export BEST_METRIC="matthews_correlation"
    elif [ $TASK_NAME = "stsb" ]; then
        export BEST_METRIC="spearmanr"
    elif [ $TASK_NAME = "rte" ]; then
        export BEST_METRIC="accuracy"
    elif [ $TASK_NAME = "qnli" ]; then
        export BEST_METRIC="accuracy"
    elif [ $TASK_NAME = "qqp" ]; then
        export BEST_METRIC="accuracy"
    fi

    finetune $TASK_NAME
else
    echo "Multi-task finetuning"
    for LR in $LR_LIST
    do
        for TASK_NAME in $TASK_LIST
        do
            export OUTPUT_DIR=${prefix}"/ckpt/roberta/finetune/"$TASK_NAME"/lr_"$LR
            export MODEL_NAME=$MODEL_NAME_PREFIX$TASK_NAME
            echo "Finetuning on "$TASK_NAME" with LR="$LR" and output_dir="$OUTPUT_DIR" and model_name="$MODEL_NAME

            if [ $TASK_NAME = "mnli" ]; then
                export BEST_METRIC="accuracy"
            elif [ $TASK_NAME = "mrpc" ]; then
                export BEST_METRIC="accuracy"
            elif [ $TASK_NAME = "sst2" ]; then
                export BEST_METRIC="accuracy"
            elif [ $TASK_NAME = "cola" ]; then
                export BEST_METRIC="matthews_correlation"
            elif [ $TASK_NAME = "stsb" ]; then
                export BEST_METRIC="spearmanr"
            elif [ $TASK_NAME = "rte" ]; then
                export BEST_METRIC="accuracy"
            elif [ $TASK_NAME = "qnli" ]; then
                export BEST_METRIC="accuracy"
            elif [ $TASK_NAME = "qqp" ]; then
                export BEST_METRIC="accuracy"
            fi

            finetune $TASK_NAME
        done
    done
fi
