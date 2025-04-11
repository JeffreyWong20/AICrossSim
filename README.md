# AI Cross-Simulation RoBERTa Experiment



## Environment Setup
Make sure the transformers library version matches the one used in the script. The script is tested with transformers==4.50.0.dev0. 

```
conda create --name aixsim python==3.13
conda activate aixsim
cd AICrossSim
python3 -m pip install -r requirements.txt
```


## Evaluation on GLUE with huggingface trainer
Evaluate RoBERTa-base on GLUE tasks. The script is modified to use the RoBERTa model and tokenizer.

```
cd ./new_compute_bench/trainer
sh new_compute_bench.sh
```

## Evaluation on GLUE with huggingface accelerate
Evaluate RoBERTa-base on GLUE tasks. The script is modified to use the RoBERTa model and tokenizer.
Accelerate provides less features as huggingface trainer but it allows us to easily make modification to the evaluation loop. This is more suitable for mase_transform.

```
cd ./new_compute_bench/no_trainer
sh new_compute_bench.sh
```


## Evaluation on GLUE with huggingface accelerate with MASE
Evaluate RoBERTa-base on GLUE tasks with mase transformation pass

```
git submodule update --init
cd ./new_compute_bench/no_trainer/submodules/mase
python3 -m pip install --upgrade pip

python3 -m pip install -e . -vvv
sh new_compute_bench.sh
```

Different transformation passes may require different configuration, evaluation code. To manage this, I’ve created a file called `access_mase.py` that stores both the transformation configuration and evaluation code.

Here’s how it works:
1. After completing a transformation in the MASE codebase, save the model weights as a `state_dict` and store the transformation configuration in `access_mase.py`. You might not need to do weight loading if your transformation is targetting the original model.
2. In `run_glue_no_trainer_mase.py`, locate the relevant `FIXME` section and update it with your own 'weight loading', 'transformation pass and configuration', and 'evaluation code'."


## Q&A

I encountered "../scipy/meson.build:216:9: ERROR: Dependency "OpenBLAS" not found, tried pkgconfig and cmake" when installing MASE
Solved by installing OpenBLAS 

```
conda install -c conda-forge openblas
```