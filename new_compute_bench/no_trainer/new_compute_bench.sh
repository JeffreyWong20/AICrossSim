#!/bin/bash
set -e  # exit on error
set -x  # print commands

# task list : [mnli qnli rte sst mrpc cola qqp stsb]
export TASK_NAME=mrpc

export model_name=JeffreyWong/roberta-base-relu-$TASK_NAME
# export model_name=JeremiahZ/roberta-base-$TASK_NAME
export use_mase="true"
export conversion_method="sta"

# Ensure output dir exists
mkdir -p ~/tmp/$TASK_NAME/eval_results

if [ "$use_mase" == "true" ]; then
  if [ "$conversion_method" == "sta" ]; then
    python run_glue_no_trainer_sta.py \
      --model_name_or_path "$model_name" \
      --task_name "$TASK_NAME" \
      --cache_dir ~/data/models \
      --data_cache_dir ~/data/datasets \
      --max_length 128 \
      --per_device_eval_batch_size 32 \
      --output_dir ~/tmp/$TASK_NAME/eval_results
  else
    python run_glue_no_trainer_mase.py \
      --model_name_or_path "$model_name" \
      --task_name "$TASK_NAME" \
      --cache_dir ~/data/models \
      --data_cache_dir ~/data/datasets \
      --max_length 128 \
      --per_device_eval_batch_size 32 \
      --output_dir ~/tmp/$TASK_NAME/eval_results
  fi
else
  python run_glue_no_trainer.py \
    --model_name_or_path "$model_name" \
    --task_name "$TASK_NAME" \
    --cache_dir ~/data/models \
    --data_cache_dir ~/data/datasets \
    --max_length 128 \
    --per_device_eval_batch_size 32 \
    --output_dir ~/tmp/$TASK_NAME/eval_results
fi
