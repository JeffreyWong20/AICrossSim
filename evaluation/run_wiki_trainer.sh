CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --nnodes=1 \
        --master_port=45000 \
        --use_env \
        run_wiki_trainer.py