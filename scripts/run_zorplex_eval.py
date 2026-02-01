#!/usr/bin/env python3
"""
Run zorplex evaluation.

Usage:
    uv run scripts/run_zorplex_eval.py --task simple
    uv run scripts/run_zorplex_eval.py --task compositional --show-samples
    uv run scripts/run_zorplex_eval.py --task multi_step --num-samples 20
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from zorplex_rl import get_spec, TASK_SPECS
from zorplex_rl.evaluate import generate_with_tools, print_result, AgenticResult


def main():
    parser = argparse.ArgumentParser(description="Agentic evaluation with tool execution")
    parser.add_argument("--task", choices=list(TASK_SPECS.keys()), required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-samples", action="store_true")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--max-turns", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get task spec
    spec_kwargs = {"seed": args.seed}
    if args.task == "compositional":
        spec_kwargs["difficulty"] = args.difficulty
    spec = get_spec(args.task, **spec_kwargs)

    # Load model
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    print(f"Loaded on {device}")

    print(f"\n{'='*60}")
    print(f"AGENTIC EVAL: {spec.name} ({spec.description})")
    print(f"Samples: {args.num_samples}, Max turns: {args.max_turns}")
    print(f"{'='*60}")

    # Run evaluation
    results: list[AgenticResult] = []

    for i in range(args.num_samples):
        task = spec.generate_task()
        result = generate_with_tools(
            model, tokenizer, spec, task, device,
            max_turns=args.max_turns,
        )
        results.append(result)

        if args.show_samples:
            print_result(result, show_trajectory=True)

    # Summary
    num_correct = sum(1 for r in results if r.is_correct)
    avg_turns = sum(len(r.turns) for r in results) / len(results)
    avg_tool_calls = sum(r.total_tool_calls for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:           {num_correct}/{args.num_samples} ({100*num_correct/args.num_samples:.1f}%)")
    print(f"Avg turns/sample:   {avg_turns:.1f}")
    print(f"Avg tool calls:     {avg_tool_calls:.1f}")
    print(f"{'='*60}")

    # Show samples if not already shown
    if not args.show_samples:
        passed = [r for r in results if r.is_correct]
        failed = [r for r in results if not r.is_correct]

        if passed:
            print(f"\n--- PASSED (up to 3) ---")
            for r in passed[:3]:
                print_result(r)

        if failed:
            print(f"\n--- FAILED (up to 5) ---")
            for r in failed[:5]:
                print_result(r)


if __name__ == "__main__":
    main()
