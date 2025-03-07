from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk
import torch
import random
import math
import evaluate

use_checkpoint = False
model_name = "yanaiela/roberta-base-epoch_2"
ckpt_path = "/home/thw20/projects/Simple-BERT-RoBERTa-Pretrain/ckpt/roberta/pretrain/base/checkpoint-4146"
# ckpt_path = "/data/models/thw20/aixsim/ckpt/roberta/pretrain/base/checkpoint-30000"
sample_size = 12800
eval_batch_size = 32


import os
torch.cuda.set_device(int(os.environ["RANK"]))
torch.cuda.empty_cache()

if use_checkpoint:
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForMaskedLM.from_pretrained(
        ckpt_path,
        cache_dir="/data/models",
        revision="main",
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        cache_dir="/data/models",
        revision="main",
    )

    

model = model.to("cuda")

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

if sample_size is not None:
    eval_dataset = load_from_disk("/data/datasets/static/Wikipedia_BookCorpus_512")['validation'].select(range(sample_size))
else:
    eval_dataset = load_from_disk("/data/datasets/static/Wikipedia_BookCorpus_512")['validation']

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

training_args = TrainingArguments(
    # output_dir="/data/models/thw20/aixsim/ckpt/roberta/eval_output",
    per_device_eval_batch_size=eval_batch_size,
    report_to="none",
)

metric = evaluate.load("accuracy", cache_dir='/data/models')

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

# Evaluate model
results = trainer.evaluate()

# Compute perplexity with numerical stability
eval_loss = results["eval_loss"]
ppl = math.exp(eval_loss) if eval_loss < 100 else float("inf")

print(f"Model Perplexity: {ppl}")
