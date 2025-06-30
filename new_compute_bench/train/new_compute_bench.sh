
# task list : [mnli	qnli rte sst mrpc cola qqp stsb]
export TASK_NAME=$1
if [ -z "$TASK_NAME" ]; then
  echo "Usage: $0 <task_name>"
  echo "Available tasks: mnli, qnli, rte, sst2, mrpc, cola, qqp, stsb"
  exit 1
fi

export model_name=JeffreyWong/roberta-base-relu-$TASK_NAME
# export model_name=JeremiahZ/roberta-base-$TASK_NAME
export use_mase=true
for lr in 1e-5 2e-5 3e-5; do
  if [ "$use_mase" = true ]; then
    python run_glue_no_trainer_mase.py \
      --model_name_or_path $model_name \
      --task_name $TASK_NAME \
      --max_length 128 \
      --per_device_eval_batch_size 32 \
      --learning_rate $lr \
      --num_train_epochs 10 \
      --output_dir /tmp/$TASK_NAME/train_results/$lr
  else
    python run_glue_no_trainer.py.bk \
      --model_name_or_path $model_name \
      --task_name $TASK_NAME \
      --max_length 128 \
      --per_device_eval_batch_size 32 \
      --output_dir /tmp/$TASK_NAME/base/train_results/$lr
  fi
done

# if [ "$use_mase" = true ]; then
#   python run_glue_no_trainer_mase.py \
#     --model_name_or_path $model_name \
#     --task_name $TASK_NAME \
#     --max_length 128 \
#     --per_device_eval_batch_size 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \
#     --output_dir /tmp/$TASK_NAME/train_results
# else
#   python run_glue_no_trainer.py.bk \
#     --model_name_or_path $model_name \
#     --task_name $TASK_NAME \
#     --max_length 128 \
#     --per_device_eval_batch_size 32 \
#     --output_dir /tmp/$TASK_NAME/base/train_results
# fi