"""
Test transformers loading and generation with Qwen 2.5 0.5B
(Fallback if vLLM has issues)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Restrict to single GPU
os.environ["HF_HUB_OFFLINE"] = "1"  # Use cached models only

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)
if device == "cpu":
    model = model.to(device)

print(f"Model loaded on {device}!")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# GSM8K problem
gsm8k_problem = """Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""

messages = [
    {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step, then give the final numerical answer."},
    {"role": "user", "content": gsm8k_problem}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(device)

print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("=" * 60)
print("PROBLEM:")
print(gsm8k_problem)
print("=" * 60)
print("MODEL RESPONSE:")
print(response)
print("=" * 60)
eggs_per_day = 16
eggs_eaten = 3
eggs_baked = 4
eggs_sold = eggs_per_day - eggs_eaten - eggs_baked
price_per_egg = 2
correct_answer = eggs_sold * price_per_egg
print(f"CORRECT ANSWER: ${correct_answer}")
print(f"  ({eggs_per_day} - {eggs_eaten} - {eggs_baked}) × ${price_per_egg} = {eggs_sold} × ${price_per_egg} = ${correct_answer}")
