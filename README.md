# AI Cross-Simulation RoBERTa Experiment



## Environment Setup
Make sure the transformers library version matches the one used in the script. The script is tested with transformers==4.50.0.dev0. 

```
conda create --name aixsim python==3.11
conda activate aixsim
cd AICrossSim
python3 -m pip install -r requirements.txt
```


## Evaluation on GLUE with huggingface accelerate
Evaluate RoBERTa-base on GLUE tasks. The script is modified to use the RoBERTa model and tokenizer.
Accelerate provides less features as huggingface trainer but it allows us to easily make modification to the evaluation loop. This is more suitable for mase_transform.

```
cd AICrossSim/new_compute_bench/no_trainer
sh new_compute_bench.sh
```

## Evaluation on GLUE with huggingface trainer
Evaluate RoBERTa-base on GLUE tasks. The script is modified to use the RoBERTa model and tokenizer.

```
cd AICrossSim/new_compute_bench/trainer
sh new_compute_bench.sh
```