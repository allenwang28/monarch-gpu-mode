import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Async RL: Building a Training Loop from Scratch

    Async RL is hard. This notebook walks through building an RL loop from the ground up, progressively disclosing Monarch concepts as we need them.

    ## When Should You RL?

    RL is a core part of the LLM pipeline - from RLHF for alignment to reasoning capabilities (o1-style thinking) to agentic training for tool use.

    But today's LLMs are already quite capable. It's not always clear you *should* RL - it's notoriously finicky. The gold standard is **verifiable rewards**: you need a task where the model has room to improve, *and* you can verify whether it succeeded.

    Let's start with the classic benchmark: **GSM8K**.
    """)
    return


@app.cell
def _():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"

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

    print(f"Loaded on {device} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")
    return device, model, tokenizer, torch


@app.cell
def _(mo):
    mo.md(r"""
    ## GSM8K: The Usual Benchmark

    GSM8K is a dataset of grade-school math problems. It's commonly used to evaluate reasoning capabilities. Let's see how our 0.5B model handles it:
    """)
    return


@app.cell
def _(device, model, tokenizer, torch):
    gsm8k_problem = """Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step, then give the final numerical answer."},
        {"role": "user", "content": gsm8k_problem}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print("PROBLEM:")
    print(gsm8k_problem)
    print("\nMODEL RESPONSE:")
    print(response)
    print(f"\n✓ CORRECT ANSWER: $18  →  (16 - 3 - 4) × $2 = 9 × $2 = $18")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Even at 0.5B parameters, Qwen 2.5 handles basic arithmetic and multi-step reasoning pretty well. **GSM8K is not a great RL target** - there's not much room for improvement.

    ## Finding a Better RL Target

    The fundamental capabilities of an agentic LLM can be summarized as:

    | Capability | Description |
    |------------|-------------|
    | **Plan** | Decide what to do next |
    | **Execute** | Carry out actions, potentially using tools |
    | **Browse** | Gather information from external sources |

    We just saw that even a small model can *plan* (multi-step reasoning works). But what about *browsing* - can it use tools effectively?

    To test this, we need a task where the model **must** use a tool to get the answer. Enter the **Zorplex Task**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Zorplex Task: A Fully Synthetic Benchmark

    **Zorplex values** are completely made-up numbers assigned to words. The model has never seen them before and cannot guess them.

    **Why synthetic benchmarks matter:**
    - Model **cannot** have memorized the answers
    - 100% verifiable - we know the ground truth
    - Clean signal for RL - no ambiguity about correctness
    - We control the difficulty

    Let's import our benchmark library:
    """)
    return


@app.cell
def _():
    from zorplex_rl import (
        TASK_SPECS,
        get_spec,
        ZORPLEX_WORDS,
        make_zorplex_table,
        generate_with_tools,
        print_result,
    )

    # Show the secret table
    table = make_zorplex_table(seed=42)
    print("Zorplex Values (seed=42):")
    for _i, (_word, _value) in enumerate(table.items()):
        if _i < 8:
            print(f"  {_word}: {_value}")
    print(f"  ... ({len(table)} total words)")

    print(f"\nRegistered task specs: {list(TASK_SPECS.keys())}")
    return TASK_SPECS, generate_with_tools, get_spec, print_result


@app.cell
def _(mo):
    mo.md(r"""
    ## Task Difficulty Levels

    We have four task types, each testing different capabilities. Let's look at what each one asks the model to do:
    """)
    return


@app.cell
def _(get_spec):
    # Show each task's system prompt
    for _task_name in ["simple", "compositional", "multi_step", "recursive"]:
        _spec = get_spec(_task_name, seed=42)
        _task = _spec.generate_task()

        print("=" * 70)
        print(f"TASK: {_task_name} - {_spec.description}")
        print("=" * 70)
        print(f"\nExample question: {_task.question}")
        print(f"Correct answer: {_task.correct_answer}")
        if _task.metadata:
            print(f"Metadata: {_task.metadata}")
        print(f"\nSystem prompt:\n{_spec.get_system_prompt(with_hint=True)}")
        print()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Running the Benchmark

    Let's evaluate the model on all task types. This uses proper tool injection - the model generates until it outputs a tool call, we execute the tool and inject the result, then continue.
    """)
    return


@app.cell
def _(TASK_SPECS, device, generate_with_tools, get_spec, model, tokenizer):
    import time

    # Run benchmark on all tasks
    NUM_SAMPLES = 10
    SEED = 42

    benchmark_results = {}
    all_latencies = []  # Track latencies across all tasks

    for bench_task_name in TASK_SPECS.keys():
        bench_spec = get_spec(bench_task_name, seed=SEED)
        results = []
        latencies = []

        for _ in range(NUM_SAMPLES):
            bench_task = bench_spec.generate_task()

            start_time = time.perf_counter()
            result = generate_with_tools(
                model, tokenizer, bench_spec, bench_task, device,
                max_turns=5,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            results.append(result)
            latencies.append(elapsed_ms)
            all_latencies.append({"task": bench_task_name, "latency_ms": elapsed_ms})

        num_correct = sum(1 for r in results if r.is_correct)
        avg_turns = sum(len(r.turns) for r in results) / len(results)
        avg_tools = sum(r.total_tool_calls for r in results) / len(results)

        benchmark_results[bench_task_name] = {
            "results": results,
            "latencies": latencies,
            "accuracy": num_correct / NUM_SAMPLES,
            "correct": num_correct,
            "total": NUM_SAMPLES,
            "avg_turns": avg_turns,
            "avg_tools": avg_tools,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "description": bench_spec.description,
        }

    # Print summary table
    print("=" * 80)
    print("ZORPLEX BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Task':<15} {'Description':<25} {'Accuracy':>10} {'Latency (ms)':>20}")
    print(f"{'':<15} {'':<25} {'':>10} {'avg':>8} {'min':>6} {'max':>6}")
    print("-" * 80)

    for bench_task_name, _data in benchmark_results.items():
        print(
            f"{bench_task_name:<15} "
            f"{_data['description']:<25} "
            f"{_data['correct']:>3}/{_data['total']:<3} ({_data['accuracy']*100:>3.0f}%) "
            f"{_data['avg_latency_ms']:>6.0f} "
            f"{_data['min_latency_ms']:>6.0f} "
            f"{_data['max_latency_ms']:>6.0f}"
        )

    print("-" * 80)
    total_correct = sum(d["correct"] for d in benchmark_results.values())
    total_samples = sum(d["total"] for d in benchmark_results.values())
    all_lats_ = [l["latency_ms"] for l in all_latencies]
    print(f"{'OVERALL':<15} {'':<25} {total_correct:>3}/{total_samples:<3} ({total_correct/total_samples*100:>3.0f}%) "
          f"{sum(all_lats_)/len(all_lats_):>6.0f} {min(all_lats_):>6.0f} {max(all_lats_):>6.0f}")
    print("=" * 80)
    return all_latencies, benchmark_results


@app.cell
def _(benchmark_results, mo):
    # Build markdown table from results
    rows = []
    for _name, _data in benchmark_results.items():
        acc_str = f"{_data['correct']}/{_data['total']} ({_data['accuracy']*100:.0f}%)"
        rows.append(f"| **{_name}** | {_data['description']} | {acc_str} |")

    table_md = "\n".join(rows)

    mo.md(f"""
    ## Results Analysis

    | Task | Description | Accuracy |
    |------|-------------|----------|
    {table_md}

    **Key observations:**

    - **Simple lookup** works perfectly - the model can call a tool and report the result
    - **Multi-step** (GETKEY → FETCH chain) also works well - sequential tool use is fine
    - **Compositional** struggles - arithmetic errors after correct tool calls
    - **Recursive** has some failures - model sometimes mimics the injection format instead of calling tools

    The model can use tools. It struggles when it needs to **compose** tool results with reasoning (arithmetic) or follow **dynamic chains** (recursive lookups).

    **This is exactly what RL can teach it.**
    """)
    return


@app.cell
def _(benchmark_results, print_result):
    # Show some example trajectories
    print("=" * 70)
    print("EXAMPLE TRAJECTORIES")
    print("=" * 70)

    # Show a compositional failure
    comp_failures = [r for r in benchmark_results["compositional"]["results"] if not r.is_correct]
    if comp_failures:
        print("\n--- Compositional failure example ---")
        print_result(comp_failures[0])

    # Show a recursive success
    rec_successes = [r for r in benchmark_results["recursive"]["results"] if r.is_correct]
    if rec_successes:
        print("\n--- Recursive success example ---")
        print_result(rec_successes[0])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why This Makes a Good RL Target

    The Zorplex benchmark is a toy example, but it illustrates the key properties we need for RL:

    1. **Clear capability gap** - The model succeeds at simple tasks but fails at compositional ones
    2. **Fully verifiable** - We know the ground truth with 100% certainty
    3. **Synthetic = no memorization** - The model must actually learn the skill
    4. **Incremental difficulty** - We can curriculum from simple → compositional → recursive

    The RL objective is straightforward:
    - **Reward = 1** if final answer matches ground truth
    - **Reward = 0** otherwise

    Now let's talk about how to build the training loop...
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## RL Primer: The 30-Second Version

    **Supervised Fine-Tuning (SFT)**: "Here's the right answer, learn to copy it."

    **Reinforcement Learning (RL)**: "Try things, I'll tell you if it worked."

    ### The MDP (Markov Decision Process)

    ```
    ┌─────────┐   action    ┌─────────────┐   reward    ┌─────────┐
    │  State  │ ──────────► │ Environment │ ──────────► │ Policy  │
    │ (context│             │  (verifier) │             │ (model) │
    │  so far)│ ◄────────── │             │ ◄────────── │         │
    └─────────┘  new state  └─────────────┘   update    └─────────┘
    ```

    The key concepts:

    | Term | In LLM Context |
    |------|----------------|
    | **Policy** | The model (maps context → next token probabilities) |
    | **State** | The conversation so far (prompt + generated tokens) |
    | **Action** | Generating token(s) - can be one token or a full response |
    | **Trajectory** | Complete rollout (prompt → response → reward) |
    | **Reward** | Score at the end (did it get the right answer?) |

    The magic of RL: the model learns from **exploration**. It doesn't need the "correct" trajectory - it discovers what works through trial and error.

    For Zorplex, this means the model can learn that:
    - Guessing values → reward = 0
    - Calling LOOKUP and waiting → reward = 1

    No one needs to *show* it the right way. It figures it out.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From MDP to Code

    We ASCII art'ed the MDP loop above. Here's what it looks like in practice - the classic **synchronous RL loop**:

    ```python
    while training:
        # 1. Generate trajectories (SLOW - this dominates)
        trajectories = model.generate(prompts)

        # 2. Score them (usually fast)
        rewards = evaluate(trajectories)

        # 3. Compute loss and update
        loss = rl_loss(trajectories, rewards)
        loss.backward()
        optimizer.step()
    ```

    This is textbook RL. The `model.generate()` step produces trajectories (policy rollouts), we score them (reward), and we update the model (policy gradient).

    But there's a problem hiding in the timing...
    """)
    return


@app.cell
def _(all_latencies, mo):
    # Calculate latency stats for the narrative
    lats_ = [l["latency_ms"] for l in all_latencies]
    lat_min, lat_max, lat_avg = min(lats_), max(lats_), sum(lats_) / len(lats_)
    lat_range = lat_max / lat_min

    mo.md(f"""
    ## The Latency Variance Problem

    Look at the latency column from our benchmark:

    - **Min**: {lat_min:.0f}ms
    - **Avg**: {lat_avg:.0f}ms
    - **Max**: {lat_max:.0f}ms
    - **Ratio**: {lat_range:.1f}x variance from fastest to slowest

    In a real RL loop, you're mixing tasks of varying difficulty. Some trajectories are short (model gets it right immediately), others are long (multiple tool calls, retries, hitting max turns).

    Since the sync loop is sequential, **something is always idle**:

    ```
    SYNC RL (batch of 4 trajectories):

    Generator 1:  ├── fast ──┤ (waiting...)
    Generator 2:  ├──── medium ────┤ (waiting...)
    Generator 3:  ├── fast ──┤ (waiting...)
    Generator 4:  ├──────────── straggler ──────────────┤
    Trainer:      (idle...........................) ├─ train ─┤
    ```

    The **long-tail stragglers** set the pace. At scale, this kills GPU utilization.
    """)
    return


@app.cell
def _(all_latencies):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    # Group latencies by task
    task_latencies = {}
    for entry in all_latencies:
        task = entry["task"]
        if task not in task_latencies:
            task_latencies[task] = []
        task_latencies[task].append(entry["latency_ms"])

    # Create overlaid KDE plot
    fig, ax = plt.subplots(figsize=(10, 4))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    all_lats = [l["latency_ms"] for l in all_latencies]
    x_range = np.linspace(min(all_lats) * 0.8, max(all_lats) * 1.1, 200)

    for i, (task, lats) in enumerate(task_latencies.items()):
        # Compute KDE
        kde = stats.gaussian_kde(lats)
        density = kde(x_range)
        ax.fill_between(x_range, density, alpha=0.3, color=colors[i], label=task)
        ax.plot(x_range, density, color=colors[i], linewidth=2)

    ax.set_xlabel("Latency (ms)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Trajectory Generation Latency Distribution by Task Type", fontsize=14)
    ax.axvline(x=sum(all_lats) / len(all_lats), color="red", linestyle="--", alpha=0.7, label="mean")
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Async Solution (and Its Trade-off)

    The fix: **decouple generation from training** with a replay buffer.

    ```
    ASYNC RL:

    Generator 1:  ├─ short ─┼── medium ──┼─ short ─┤
    Generator 2:  ├── medium ──┼─ short ─┼──── long ────┤
    Generator 3:  ├──── long ────┼─ short ─┼── medium ──┤
    Generator 4:  ├─ short ─┼──── long ────┼─ short ─┤
                        ↓         ↓         ↓
    Buffer:       ══════════════════════════════════════
                        ↓         ↓         ↓
    Trainer:      ├─ train ─┼─ train ─┼─ train ─┼─ train ─┤
    ```

    Generators produce trajectories as fast as they can, dropping them in a buffer. The trainer consumes from the buffer independently. Variance doesn't matter - the buffer absorbs it.

    ### The Off-Policy Trade-off

    But there's a catch: by the time the trainer uses a trajectory, the model weights may have changed. The trajectory was generated by an **old policy** - this is called **off-policy** training.

    | Approach | On-Policy | Off-Policy |
    |----------|-----------|------------|
    | Trajectory freshness | Always from current weights | May be stale |
    | GPU utilization | Low (waiting) | High (async) |
    | Learning stability | More stable | Requires correction |

    Modern algorithms (like GRPO, PPO with importance sampling) handle off-policy data with correction terms. The systems efficiency gain is worth it.

    **This is the systems challenge that Monarch solves** - distributed actors generating trajectories, feeding a central trainer, with efficient weight synchronization via RDMA.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interactive Visualization: Sync vs Async

    Use the sliders to see how different configurations affect GPU utilization:
    """)
    return


@app.cell
def _(mo):
    gen_time = mo.ui.slider(100, 2000, value=800, label="Generation time (ms)")
    train_time = mo.ui.slider(50, 500, value=200, label="Training time (ms)")
    sync_time = mo.ui.slider(10, 200, value=50, label="Weight sync time (ms)")
    num_generators = mo.ui.slider(1, 8, value=4, label="Number of generators")

    mo.vstack([gen_time, train_time, sync_time, num_generators])
    return gen_time, num_generators, sync_time, train_time


@app.cell
def _(gen_time, mo, num_generators, sync_time, train_time):
    total_sync = gen_time.value + train_time.value + sync_time.value
    sync_util = train_time.value / total_sync * 100

    async_gen_throughput = gen_time.value / num_generators.value
    total_async = max(async_gen_throughput, train_time.value) + sync_time.value
    async_util = train_time.value / total_async * 100

    mo.md(f"""
    ### Results

    | Metric | Sync RL | Async RL |
    |--------|---------|----------|
    | Trainer utilization | {sync_util:.1f}% | {async_util:.1f}% |
    | Steps/second | {1000/total_sync:.2f} | {1000/total_async:.2f} |

    **Speedup: {total_sync/total_async:.1f}x**

    With {num_generators.value} generators, async RL keeps the trainer {async_util:.0f}% utilized vs {sync_util:.0f}% for sync.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Next Steps

    We've established:
    1. **The task** - Zorplex benchmark with clear capability gaps
    2. **The reward** - Binary correctness signal
    3. **The motivation** - Async RL for better GPU utilization

    Next, we'll build the actual training loop using Monarch:
    - Generator actors that produce rollouts
    - A trainer that consumes them
    - Weight synchronization between them
    - Fault tolerance when things go wrong
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
