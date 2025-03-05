python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --nnodes=1 \
        --master_port=48000 \
        --use_env \
        run_wiki_trainer.py