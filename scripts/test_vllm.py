"""
Test vLLM loading and generation with Qwen 2.5 0.5B
"""
import os

# Restrict to single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_USE_V1"] = "0"  # Use legacy engine if v1 has issues
os.environ["HF_HUB_OFFLINE"] = "1"  # Use cached models only

from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading {MODEL_NAME}...")
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    dtype="float16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.3,
)
print("Model loaded!")

# GSM8K problem
gsm8k_problem = """Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""

prompt = f"""<|im_start|>system
You are a helpful assistant. Solve the math problem step by step, then give the final numerical answer.<|im_end|>
<|im_start|>user
{gsm8k_problem}<|im_end|>
<|im_start|>assistant
"""

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
)

print("Generating...")
outputs = llm.generate([prompt], sampling_params)
response = outputs[0].outputs[0].text

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
