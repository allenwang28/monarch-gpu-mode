#!/usr/bin/env python3
"""
Evaluate a model on all registered zorplex task specs.

Usage:
    uv run scripts/run_zorplex_benchmark.py
    uv run scripts/run_zorplex_benchmark.py --num-samples 20
    uv run scripts/run_zorplex_benchmark.py --model meta-llama/Llama-3.2-1B-Instruct
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from zorplex_rl import TASK_SPECS, get_spec
from zorplex_rl.evaluate import generate_with_tools, AgenticResult


def main():
    parser = argparse.ArgumentParser(description="Benchmark all zorplex tasks")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Model name or path")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model once
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    print(f"Loaded on {device}\n")

    # Results storage
    results_table = []

    # Evaluate each task spec
    for task_name in TASK_SPECS.keys():
        print(f"Evaluating {task_name}...", end=" ", flush=True)

        spec = get_spec(task_name, seed=args.seed)
        results: list[AgenticResult] = []

        for i in range(args.num_samples):
            task = spec.generate_task()
            result = generate_with_tools(
                model, tokenizer, spec, task, device,
                max_turns=args.max_turns,
            )
            results.append(result)

        # Compute metrics
        num_correct = sum(1 for r in results if r.is_correct)
        accuracy = num_correct / args.num_samples
        avg_turns = sum(len(r.turns) for r in results) / len(results)
        avg_tool_calls = sum(r.total_tool_calls for r in results) / len(results)

        results_table.append({
            "task": task_name,
            "accuracy": accuracy,
            "correct": num_correct,
            "total": args.num_samples,
            "avg_turns": avg_turns,
            "avg_tool_calls": avg_tool_calls,
            "description": spec.description,
        })

        print(f"{num_correct}/{args.num_samples} ({accuracy*100:.0f}%)")

    # Print summary table
    print("\n" + "=" * 70)
    print("ZORPLEX BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Samples per task: {args.num_samples}")
    print(f"Max turns: {args.max_turns}")
    print("=" * 70)
    print()
    print(f"{'Task':<15} {'Description':<25} {'Accuracy':>10} {'Turns':>8} {'Tools':>8}")
    print("-" * 70)

    for row in results_table:
        print(
            f"{row['task']:<15} "
            f"{row['description']:<25} "
            f"{row['correct']:>3}/{row['total']:<3} ({row['accuracy']*100:>3.0f}%) "
            f"{row['avg_turns']:>6.1f} "
            f"{row['avg_tool_calls']:>7.1f}"
        )

    print("-" * 70)

    # Overall stats
    total_correct = sum(r["correct"] for r in results_table)
    total_samples = sum(r["total"] for r in results_table)
    overall_accuracy = total_correct / total_samples

    print(f"{'OVERALL':<15} {'':<25} {total_correct:>3}/{total_samples:<3} ({overall_accuracy*100:>3.0f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
