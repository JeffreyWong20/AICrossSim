# AI Cross-Simulation RoBERTa Experiment



## Environment Setup
Make sure the transformers library version matches the one used in the script. The script is tested with transformers==4.50.0.dev0. 

```
conda create --name aixsim python==3.11
conda activate aixsim
cd AICrossSim
python3 -m pip install -r requirements.txt
```


## Fine-tuning on GLUE
finetune_base.sh contains the script to fine-tune RoBERTa-base on GLUE tasks. The script is based on the run_glue.py script from the Hugging Face Transformers library. The script is modified to use the RoBERTa model and tokenizer.

```
cd AICrossSim
sh finetune_base.sh
```