import os
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import glob
import subprocess

MODEL_NAME = "Maykeye/TinyLLama-v0"
OUTPUT_DIR = "./tinyllama-epstein-finetuned"

# Find latest checkpoint or final model
checkpoints = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")), reverse=True)
if checkpoints:
    model_path = checkpoints[0]
    print(f"Loading from latest checkpoint: {model_path}")
else:
    model_path = OUTPUT_DIR
    print(f"Loading from final model: {model_path}")

# Device setup
if torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS GPU.")
else:
    device = torch.device("cpu")
    print("Using CPU.")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": device},
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, model_path)

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
except:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n" + "="*60)
subprocess.run("clear",text=True,shell=True)
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
print("="*60)

while True:
    user_input = input("""
You: """)
    if user_input.lower() == "quit":
        break
    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(user_input):].strip()
    for word in response.split():
        print(word, end=' ', flush=True)
        time.sleep(0.05)
    print()

print("Goodbye!")
