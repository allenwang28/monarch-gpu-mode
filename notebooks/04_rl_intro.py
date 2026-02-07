import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # RL at Scale: The Systems Problem

    You've learned how Monarch organizes distributed work: actors communicate
    through endpoints, meshes provide structured parallelism, and supervision
    trees keep things running when GPUs fail. Now the question: what do you
    **build** with all of this?

    Increasingly, **RL** is one of the most natural use cases. From RLHF for
    alignment to reasoning capabilities (o1-style thinking) to agentic tool
    use, RL is a core part of the modern LLM pipeline. And RL at scale is
    fundamentally a distributed systems problem — generators producing
    rollouts, trainers consuming them, weights flowing between them — that
    maps naturally to the actor model.

    This notebook establishes the groundwork:

    1. **The task** — identifying a synthetic benchmark where the model has room to improve
    2. **The reward** — fully verifiable, no ambiguity
    3. **The bottleneck** — why synchronous RL wastes GPUs
    4. **The architecture** — async RL, and why it maps to actors

    But first: **when does RL actually need distributed systems?** Let's build
    intuition.
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
    ## GSM8K: A Classic RL Target

    GSM8K (grade-school math) is one of the most popular RL benchmarks for
    LLMs. It's a great RL target because it has **verifiable rewards** (we can
    check the numerical answer) and the model generates **reasoning traces**
    we can inspect. Let's see what that looks like:
    """)
    return


@app.cell
def _(device, model, tokenizer, torch):
    # Fix seed for reproducible output
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

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
    print("="*10)
    print("\nThe correct answer should be: $18  →  (16 - 3 - 4) × $2 = 9 × $2 = $18")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Even at 0.5B parameters, Qwen 2.5 can generate a convincing looking reasoning trace
     — but it doesn't always produce the right answer.

    That's actually a nice property: the fact
    that the model *sometimes* gets answers right gives us confidence that
    further training can elicit more of this behavior. If it never succeeded,
    RL would have no positive signal to reinforce.

    GSM8K shows the basic RL setup: the model generates a reasoning trace,
    we check the final answer, and we get a binary reward. This is the
    pattern behind DeepSeek-R1 and similar work.

    But RL tasks come in many flavors. Tool-use tasks are more complex — the
    model must call external APIs, chain results, and do arithmetic on
    retrieved values. These tasks also produce **variable-length
    trajectories**, which will matter when we talk about the systems
    architecture.

    For this tutorial, we'll use a fully synthetic tool-use benchmark where
    we control everything — difficulty, ground truth, and latency profile.

    Enter the **Zorplex Task**.
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
    ### How Tool Injection Works

    The `generate_with_tools` function implements an **agentic loop** — generate
    until the model outputs a tool call, execute it, inject the result, and
    continue:

    ```
    ┌─────────────────────────────────────────────────────────────┐
    │  1. Generate text until tool call detected                  │
    │     └─ StoppingCriteria regex-matches LOOKUP[...]           │
    │                                                             │
    │  2. Parse and execute the tool call                         │
    │     └─ LOOKUP[banana] → 42                                  │
    │                                                             │
    │  3. Inject result into context: "[Result: 42]"              │
    │                                                             │
    │  4. Repeat until no tool calls or max turns                 │
    └─────────────────────────────────────────────────────────────┘
    ```

    The model sees tool-use examples in the system prompt, learns the format,
    and gets results injected back. Let's see what each difficulty level
    asks it to do:
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Task Difficulty Levels
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
def _(
    TASK_SPECS,
    device,
    generate_with_tools,
    get_spec,
    model,
    tokenizer,
    torch,
):
    import time
    import random

    # Run benchmark on all tasks
    NUM_SAMPLES = 10
    SEED = 42

    # Set seeds for reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    benchmark_results = {}
    all_latencies = []  # Track latencies across all tasks

    for bench_task_name in TASK_SPECS.keys():
        bench_spec = get_spec(bench_task_name, seed=SEED)
        results = []
        latencies = []

        print(f"Running {bench_task_name}...", end=" ", flush=True)
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
        print(f"{num_correct}/{NUM_SAMPLES} correct")
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
    print("=" * 60)
    print("ZORPLEX BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Task':<15} {'Description':<25} {'Accuracy':>12}")
    print("-" * 60)

    for bench_task_name, _data in benchmark_results.items():
        print(
            f"{bench_task_name:<15} "
            f"{_data['description']:<25} "
            f"{_data['correct']:>3}/{_data['total']:<3} ({_data['accuracy']*100:>3.0f}%)"
        )

    print("-" * 60)
    total_correct = sum(d["correct"] for d in benchmark_results.values())
    total_samples = sum(d["total"] for d in benchmark_results.values())
    print(f"{'OVERALL':<15} {'':<25} {total_correct:>3}/{total_samples:<3} ({total_correct/total_samples*100:>3.0f}%)")
    print("=" * 60)
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

    Accuracy is lower than you might expect — and that's largely because of the
    `[ANSWER]` format requirement. The model can often get *right value* but
    fails to emit it in the required format. This is actually good news for RL:
    the capability is there, it just needs to be reinforced.

    Let's look at concrete trajectories to understand the failure modes.
    """)
    return


@app.cell
def _(benchmark_results, print_result):
    import re as _re

    # Classify results by failure mode
    successes = []
    format_failures = []  # Right value, wrong format
    tool_spam = []        # Excessive tool calls without progress
    wrong_answer = []     # Wrong computation or value

    for _task_name, _data in benchmark_results.items():
        for _r in _data["results"]:
            if _r.is_correct:
                successes.append((_task_name, _r))
            else:
                # Check if the correct answer appears in the trajectory but wasn't extracted
                correct_str = str(_r.task.correct_answer)
                has_correct_value = correct_str in _r.final_text

                # Check for [ANSWER] tag presence
                has_answer_tag = bool(_re.search(r'\[ANSWER\]', _r.final_text))

                if not has_answer_tag:
                    format_failures.append((_task_name, _r))
                elif _r.total_tool_calls > 3:
                    tool_spam.append((_task_name, _r))
                else:
                    wrong_answer.append((_task_name, _r))

    # --- Successes ---
    print("=" * 70)
    print(f"SUCCESS EXAMPLES ({len(successes)} total)")
    print("=" * 70)
    for _task_name, _r in successes[:2]:
        print(f"\n--- {_task_name} ---")
        print_result(_r)

    # --- Format failures ---
    print("\n" + "=" * 70)
    print(f"FAILURE MODE 1: Wrong format ({len(format_failures)} total)")
    print("The model never emitted [ANSWER].")
    print("=" * 70)
    for _task_name, _r in format_failures[:2]:
        print(f"\n--- {_task_name} ---")
        print_result(_r)

    # --- Tool spam ---
    if tool_spam:
        print("\n" + "=" * 70)
        print(f"FAILURE MODE 2: Tool spam ({len(tool_spam)} total)")
        print("Excessive tool calls without making progress.")
        print("=" * 70)
        for _task_name, _r in tool_spam[:2]:
            print(f"\n--- {_task_name} ---")
            print_result(_r)

    # --- Wrong answer ---
    if wrong_answer:
        print("\n" + "=" * 70)
        print(f"FAILURE MODE 3: Wrong answer ({len(wrong_answer)} total)")
        print("Incorrect computation or value extraction.")
        print("=" * 70)
        for _task_name, _r in wrong_answer[:2]:
            print(f"\n--- {_task_name} ---")
            print_result(_r)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Successes:              {len(successes)}")
    print(f"  Right value, bad format: {len(format_failures)}")
    print(f"  Tool spam:              {len(tool_spam)}")
    print(f"  Wrong answer:           {len(wrong_answer)}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From Failure Modes to RL Rewards

    The trajectories above reveal three distinct failure modes, and each maps
    to a reward signal:

    | Failure Mode | What Happens | RL Signal |
    |---|---|---|
    | **Wrong format** | Model doesn't emit `[ANSWER]` | Reward format compliance |
    | **Tool spam** | Excessive LOOKUPs without progress | Penalize unnecessary tool calls |
    | **Wrong answer** | Incorrect arithmetic or value extraction | Binary correctness reward (0 or 1) |

    The key insight: **the model already has most of the capability**. It can
    call tools, follow redirects, and often arrives at the right value. What
    it lacks is the discipline to format answers correctly and compose results
    reliably. These are exactly the behaviors RL can reinforce.

    Now let's talk about how to build the training loop...
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From Task to Training Loop

    Unlike supervised fine-tuning ("here's the right answer, copy it"), RL
    learns from **exploration** — the model tries things and gets a reward
    signal. For Zorplex, this means it can discover on its own that guessing
    values gives reward 0, while calling LOOKUP gives reward 1. No one needs
    to show it the right trajectory.

    **Note on reward design:** The binary reward (correct=1, incorrect=0) is
    simple but sparse. In practice, you might add **reward shaping** —
    penalties for excessive tool calls (compositional tasks tend to spam
    LOOKUP), bonuses for correct output formatting. We'll explore shaped
    rewards in NB07 when we wire up the full RL loop.

    The standard **synchronous RL loop** looks like this:

    ```python
    # Each step: generate on all GPUs, wait, then train
    while training:
        trajectories = generators.generate(prompts)  # SLOW — dominates runtime
        rewards = evaluate(trajectories)              # fast — check answer
        loss = rl_loss(trajectories, rewards)          # policy gradient update
        loss.backward()
        optimizer.step()
    ```

    Each generator produces one trajectory per step. We wait for **all** of
    them to finish before training. Spot the problem?
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

    Look at the latencies from our benchmark:

    - **Min**: {lat_min:.0f}ms
    - **Avg**: {lat_avg:.0f}ms
    - **Max**: {lat_max:.0f}ms
    - **Ratio**: {lat_range:.1f}x variance from fastest to slowest

    In a real RL loop, you're mixing tasks of varying difficulty. Some
    trajectories finish fast, others take multiple tool calls and hit max
    turns. In the sync loop, the trainer sits idle during generation, and
    generators sit idle during training — the slowest trajectory sets the
    pace for the entire batch:
    """)
    return


@app.cell
def _():
    def make_sync_plot():
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # Sync RL visualization - showing straggler problem
        fig, ax = plt.subplots(figsize=(10, 3))

        # Data: (row, start, duration, color)
        data = [
            (0, 0, 3, "#66c2a5"),   # Gen 1: fast
            (1, 0, 5, "#66c2a5"),   # Gen 2: medium
            (2, 0, 3, "#66c2a5"),   # Gen 3: fast
            (3, 0, 10, "#fc8d62"), # Gen 4: straggler (different color)
            (4, 0, 10, "#d3d3d3"), # Trainer: idle
            (4, 10, 3, "#e78ac3"), # Trainer: train
        ]

        for row, start, duration, color in data:
            ax.barh(row, duration, left=start, height=0.6, color=color, edgecolor="black", linewidth=0.5)

        # Add "idle" and "train" labels
        ax.text(5, 4, "idle", ha="center", va="center", fontsize=9, color="gray")
        ax.text(11.5, 4, "train", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

        ax.set_yticks(range(5))
        ax.set_yticklabels(["Gen 1", "Gen 2", "Gen 3", "Gen 4", "Trainer"])
        ax.set_xlabel("Time (ms)")
        ax.set_title("Sync RL - Waiting for Stragglers")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

        # Legend
        patches = [
            mpatches.Patch(color="#66c2a5", label="Fast/Medium"),
            mpatches.Patch(color="#fc8d62", label="Straggler"),
            mpatches.Patch(color="#d3d3d3", label="Idle"),
            mpatches.Patch(color="#e78ac3", label="Train"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=8)

        plt.tight_layout()
        return fig

    make_sync_plot()
    return


@app.cell
def _(mo):
    mo.md(r"""
    The **long-tail stragglers** set the pace. At scale, this kills GPU utilization.
    """)
    return


@app.cell
def _(all_latencies):
    def make_kde_plot(latencies):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats

        # Group latencies by task
        task_latencies = {}
        for entry in latencies:
            task = entry["task"]
            if task not in task_latencies:
                task_latencies[task] = []
            task_latencies[task].append(entry["latency_ms"])

        # Create overlaid KDE plot
        fig, ax = plt.subplots(figsize=(10, 4))

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        all_lats = [l["latency_ms"] for l in latencies]
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
        return fig

    make_kde_plot(all_latencies)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Async Solution (and Its Trade-off)

    The fix: **decouple generation from training** with a replay buffer.
    """)
    return


@app.cell
def _():
    def make_async_plot():
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # Async RL visualization - showing continuous utilization
        fig, ax = plt.subplots(figsize=(10, 3.5))

        # Colors for different trajectory lengths
        color_map = {"short": "#66c2a5", "medium": "#8da0cb", "long": "#fc8d62"}

        # Data: (row, start, duration, type)
        data = [
            # Gen 1: short, medium, short
            (0, 0, 2, "short"), (0, 2, 4, "medium"), (0, 6, 2, "short"),
            # Gen 2: medium, short, long
            (1, 0, 4, "medium"), (1, 4, 2, "short"), (1, 6, 6, "long"),
            # Gen 3: long, short, medium
            (2, 0, 5, "long"), (2, 5, 2, "short"), (2, 7, 4, "medium"),
            # Gen 4: short, long, short
            (3, 0, 2, "short"), (3, 2, 6, "long"), (3, 8, 2, "short"),
            # Buffer (visual indicator)
            (4, 0, 12, "#e0e0e0"),
            # Trainer: continuous training
            (5, 0, 3, "#e78ac3"), (5, 3, 3, "#e78ac3"), (5, 6, 3, "#e78ac3"), (5, 9, 3, "#e78ac3"),
        ]

        for item in data:
            row, start, duration = item[0], item[1], item[2]
            if isinstance(item[3], str) and item[3] in color_map:
                color = color_map[item[3]]
            else:
                color = item[3]
            ax.barh(row, duration, left=start, height=0.6, color=color, edgecolor="black", linewidth=0.5)

        # Buffer label
        ax.text(6, 4, "Buffer", ha="center", va="center", fontsize=9, color="gray")

        ax.set_yticks(range(6))
        ax.set_yticklabels(["Gen 1", "Gen 2", "Gen 3", "Gen 4", "Buffer", "Trainer"])
        ax.set_xlabel("Time (ms)")
        ax.set_title("Async RL - Buffer Absorbs Variance")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

        # Legend
        patches = [
            mpatches.Patch(color="#66c2a5", label="Short"),
            mpatches.Patch(color="#8da0cb", label="Medium"),
            mpatches.Patch(color="#fc8d62", label="Long"),
            mpatches.Patch(color="#e78ac3", label="Train"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=8)

        plt.tight_layout()
        return fig

    make_async_plot()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Generators produce trajectories as fast as they can, dropping them into a
    **replay buffer**. The trainer consumes from the buffer independently.
    The buffer absorbs more of the performance impact of the variation in generation time latencies.

    ### Why Async Is Worth It (Beyond Stragglers)

    The straggler problem is the most obvious win, but async RL has deeper
    systems benefits:

    - **Independent scaling** — tune the generator-to-trainer ratio without
      changing code. Adding generators becomes changing a number in your actor spawn call.
    - **Heterogeneous hardware** — use inference-optimized GPUs for generation
      and training-optimized GPUs (with fast interconnect) for gradient updates.
    - **Fault isolation** — a crashed generator is just one fewer producer. The
      buffer and trainer are unaffected. Compare this to sync RL, where one
      crash loses the entire batch.

    ### The Off-Policy Trade-off

    There's a catch: by the time the trainer uses a trajectory, the model
    weights may have changed. The trajectory was generated by an **old
    policy** — this is called **off-policy** training.

    | Aspect | Sync (on-policy) | Async (off-policy) |
    |--------|------------------|--------------------|
    | Trajectory freshness | Always current weights | May be stale |
    | GPU utilization | Low (waiting) | High (continuous) |
    | Learning stability | More stable | Requires bounded staleness |

    In practice, systems keep the staleness bounded — trajectories more than
    a few gradient steps old get evicted from the buffer. PPO handles mild
    off-policy data through importance sampling ratios (π_new / π_old),
    clipped to prevent large policy updates. The systems efficiency gain
    is worth it.

    Strictly speaking, algorithms like PPO and GRPO are meant to be
    **on-policy** — they assume trajectories come from the current weights.
    But using clipped importance sampling ratios (π_new / π_old) helps
    tolerate mild staleness empirically, allowing async systems to trade
    a small amount of policy freshness for significantly better throughput.

    ### Why This Maps to Actors

    Look at the async architecture: generators, a buffer, a trainer — each
    is an independent process that communicates via messages. This **is** the
    actor model from notebooks 1-2. Monarch gives us:

    - **Actors with endpoints** for defining remote generators, buffer, and trainer
    - **Fault tolerance** (NB03) for keeping generators alive when GPUs fail
    - **Services** (NB05) for managing pools of generators with health tracking
    - **RDMA weight sync** (NB06) for moving hundreds of MB without blocking
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interactive Visualization: Sync vs Async

    Adjust the parameters to see how generation variance affects sync vs async RL.

    **Note:** We assume batch size = number of generators (one trajectory per generator per batch).
    """)
    return


@app.cell
def _(mo):
    num_generators = mo.ui.slider(2, 6, value=4, label="Number of generators")
    mean_gen_time = mo.ui.slider(100, 500, value=200, label="Mean generation time (ms)")
    variance = mo.ui.slider(0.1, 2.0, value=0.8, step=0.1, label="Variance (coefficient of variation)")
    train_time = mo.ui.slider(50, 300, value=100, label="Training time (ms)")
    weight_sync_time = mo.ui.slider(10, 100, value=30, label="Weight sync time (ms)")
    num_batches = mo.ui.slider(2, 5, value=3, label="Batches to simulate")

    mo.vstack([num_generators, mean_gen_time, variance, train_time, weight_sync_time, num_batches])
    return (
        mean_gen_time,
        num_batches,
        num_generators,
        train_time,
        variance,
        weight_sync_time,
    )


@app.cell
def _(
    mean_gen_time,
    mo,
    num_batches,
    num_generators,
    train_time,
    variance,
    weight_sync_time,
):
    def make_interactive_comparison(n_gens_val, mean_val, var_val, train_val, sync_val, n_batch_val):
        import random
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import math

        # Seed based on slider values so different configs give different samples
        random.seed(42 + int(mean_val) + int(var_val * 100) + n_gens_val + n_batch_val)

        # Sample generation times from log-normal distribution
        def sample_gen_times(mean, cv, n):
            if cv < 0.1:
                return [mean] * n
            sigma = math.sqrt(math.log(1 + cv**2))
            mu = math.log(mean) - sigma**2 / 2
            return [random.lognormvariate(mu, sigma) for _ in range(n)]

        # Generate times for each generator across batches
        gen_times = {}
        for g in range(n_gens_val):
            gen_times[g] = sample_gen_times(mean_val, var_val, n_batch_val)

        # Color palette for batches
        batch_colors = plt.cm.Set2(range(n_batch_val))

        # ===== SYNC SIMULATION =====
        sync_bars = []
        sync_time = 0

        for batch in range(n_batch_val):
            batch_times = [gen_times[g][batch] for g in range(n_gens_val)]
            max_time = max(batch_times)

            for g in range(n_gens_val):
                t = gen_times[g][batch]
                sync_bars.append((g, sync_time, t, batch_colors[batch]))

            train_start = sync_time + max_time
            sync_bars.append((n_gens_val, train_start, train_val, "crimson"))
            # Weight sync after training
            sync_bars.append((n_gens_val + 1, train_start + train_val, sync_val, "#9467bd"))
            sync_time = train_start + train_val + sync_val

        total_sync_time = sync_time

        # ===== ASYNC SIMULATION =====
        # In async: generators run continuously, pulling weights when new version available
        # Weight sync is generator-side: after each batch, if trainer has new weights, pull them
        async_bars = []

        # Track when each training step completes (new weights available)
        train_complete_times = []
        t = 0
        while t < sum(gen_times[0]) * 1.5:  # Run trainer long enough
            t += train_val
            train_complete_times.append(t)

        # Each generator: generate, then sync if new weights available
        for g in range(n_gens_val):
            t = 0
            last_synced_version = -1
            for batch in range(n_batch_val):
                # Check if new weights available (trainer completed a step since last sync)
                new_version = sum(1 for tc in train_complete_times if tc <= t) - 1
                if new_version > last_synced_version and new_version >= 0:
                    # Pull new weights (sync time)
                    async_bars.append((g, t, sync_val, "#9467bd"))
                    t += sync_val
                    last_synced_version = new_version

                # Generate
                duration = gen_times[g][batch]
                async_bars.append((g, t, duration, batch_colors[batch]))
                t += duration

        total_gen_time = max(
            sum(gen_times[g]) + sync_val * n_batch_val  # Rough upper bound
            for g in range(n_gens_val)
        )

        # Trainer just runs continuously
        t = 0
        train_batch = 0
        while t < total_gen_time:
            async_bars.append((n_gens_val, t, train_val, "crimson"))
            t += train_val
            train_batch += 1

        total_async_time = max(
            max(bar[1] + bar[2] for bar in async_bars if bar[0] < n_gens_val),  # Max generator end
            t  # Trainer end
        )

        # ===== PLOTTING =====
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5 + n_gens_val * 0.3))
        sync_row_labels = [f"Gen {i+1}" for i in range(n_gens_val)] + ["Trainer", "Wt Sync"]
        async_row_labels = [f"Gen {i+1}" for i in range(n_gens_val)] + ["Trainer"]

        # Sync RL plot
        ax = axes[0]
        for row, start, duration, color in sync_bars:
            ax.barh(row, duration, left=start, height=0.6, color=color, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(sync_row_labels)))
        ax.set_yticklabels(sync_row_labels)
        ax.set_xlabel("Time (ms)")
        ax.set_title(f"Sync RL (total: {total_sync_time:.0f}ms)")
        ax.set_xlim(0, max(total_sync_time, total_async_time) * 1.05)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

        # Async RL plot
        ax = axes[1]
        for row, start, duration, color in async_bars:
            ax.barh(row, duration, left=start, height=0.6, color=color, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(async_row_labels)))
        ax.set_yticklabels(async_row_labels)
        ax.set_xlabel("Time (ms)")
        ax.set_title(f"Async RL (total: {total_async_time:.0f}ms)")
        ax.set_xlim(0, max(total_sync_time, total_async_time) * 1.05)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

        # Legend
        patches = [mpatches.Patch(color=batch_colors[i], label=f"Batch {i+1}") for i in range(n_batch_val)]
        patches.append(mpatches.Patch(color="crimson", label="Train"))
        patches.append(mpatches.Patch(color="#9467bd", label="Wt Sync"))
        axes[1].legend(handles=patches, loc="upper right", fontsize=8)

        plt.tight_layout()

        # Calculate utilization
        total_train_sync = n_batch_val * train_val
        sync_util = total_train_sync / total_sync_time * 100
        async_util = min(100, train_batch * train_val / total_async_time * 100)

        return fig, total_sync_time, total_async_time, sync_util, async_util

    fig, total_sync, total_async, sync_util, async_util = make_interactive_comparison(
        num_generators.value, mean_gen_time.value, variance.value,
        train_time.value, weight_sync_time.value, num_batches.value
    )

    mo.vstack([
        fig,
        mo.md(f"""
    | Metric | Sync RL | Async RL |
    |--------|---------|----------|
    | Total time | {total_sync:.0f}ms | {total_async:.0f}ms |
    | Trainer utilization | {sync_util:.0f}% | {async_util:.0f}% |
    | **Speedup** | - | **{total_sync/total_async:.1f}x** |

    **Try increasing the variance slider** to see how stragglers hurt sync RL while async handles them gracefully.
        """)
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Next Steps

    We've established the task (Zorplex), the reward (binary correctness),
    and the architecture (async RL with actors). Now we need to build it.

    The next three notebooks each tackle one piece:

    - **NB05 — Services**: How do you manage a pool of generator actors?
      Round-robin routing, health tracking, and discovery — so the trainer
      can find generators without hardcoding addresses.
    - **NB06 — RDMA Weight Sync**: After each training step, generators
      need fresh weights. But how do you move hundreds of MB without blocking
      training? RDMA separates the control plane (tiny handle message) from
      the data plane (bulk transfer).
    - **NB07 — Async RL E2E**: Wire it all together — generators, buffer,
      trainer, weight sync — into a working async RL loop on Zorplex.
    """)
    return


if __name__ == "__main__":
    app.run()
