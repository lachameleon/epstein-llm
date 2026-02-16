# -*- coding: utf-8 -*-

"""
Fully automatic TinyLLama-v0 fine-tuning on the
'KillerShoaib/Jeffrey-Epstein-Emails-From-Epstein-Files' dataset.
- Works on macOS (MPS or CPU).
- Automatically saves checkpoints and resumes if interrupted.
- Uses LoRA for parameter-efficient training.
- Preprocesses and tokenizes dataset automatically.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import subprocess
import glob

# ==============================
# RAM Limit (6GB)
# ==============================
# ram_limit = 6 * 1024 * 1024 * 1024
# resource.setrlimit(resource.RLIMIT_AS, (ram_limit, ram_limit))
# print(f"Set virtual memory limit to {ram_limit / (1024**3):.0f}GB")

# ==============================
# Clear Terminal
# ==============================
subprocess.run("clear", shell=True, text=True)

print('''
          _
         /_'. _
       _   \ / '-.
      < ``-.;),--'`
       '--.</()`--.
         / |/-/`'._\\
         |/ |=|
            |_|
       ~`   |-| ~~      ~
   ~~  ~~ __|=|__   ~~
 ~~   .-'`  |_|  ``""-._   ~~
  ~~.'      |=|    O    '-.  ~
    |      `"""`  <|\\      \\   ~
~   \\              |\\      | ~~
 jgs '-.__.--._    |/   .-'
          ~~   `--...-'`    ~~
  ~~         ~          ~
         ~~         ~~     ~
''')

print("Welcome to the Island!")

MODEL_NAME = "Maykeye/TinyLLama-v0"
DATASET_NAME = "KillerShoaib/Jeffrey-Epstein-Emails-From-Epstein-Files"
OUTPUT_DIR = "./tinyllama-epstein-finetuned"
MAX_LENGTH = 512
BATCH_SIZE = 1
GRAD_ACCUM = 16
EPOCHS = 10
LEARNING_RATE = 5e-4
WARMUP_RATIO = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

if torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS GPU (Metal) for training.")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU. Training will be slower.")


print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

if "train" in dataset:
    dataset = dataset["train"]

print("Dataset info:")
print(dataset)
print("Columns:", dataset.column_names)
print("First example:", dataset[0])


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def preprocess(example):
    body = example.get("body", "")
    subject = example.get("subject", "")
    from_name = example.get("from_name", "")
    to_field = example.get("to", "")
    date = example.get("date", "")

    if not body:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    text = f"""Subject: {subject}
From: {from_name}
To: {to_field}
Date: {date}

{body}
"""

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
    batched=False
)

tokenized_dataset = tokenized_dataset.filter(
    lambda x: len(x["input_ids"]) > 0
)


print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": device},
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)


lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=5,
    optim="adamw_torch",
    fp16=False,
    bf16=False,
    report_to="none",
    remove_unused_columns=False,
    max_grad_norm=1.0,
    gradient_checkpointing=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Resume From Checkpoint
checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
if checkpoints:
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]), reverse=True)

latest_checkpoint = checkpoints[0] if checkpoints else None

if latest_checkpoint:
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("No checkpoint found, training from scratch.")
    trainer.train()

# Final Save

print("Saving final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model saved to {OUTPUT_DIR}")
print("Training complete! Island conquered! 🏝️")
