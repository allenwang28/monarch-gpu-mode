#!/usr/bin/env python3
"""
Notebook 07: Closing the Loop - Async RL Training

In Notebook 04, we saw that Qwen 0.5B struggles with compositional Zorplex tasks.
Now we close the loop: train the model using async RL with Monarch.

This notebook demonstrates:
- Building RL actors (Trainer, Generator, ReplayBuffer)
- Running concurrent generation and training
- Weight synchronization between actors (circular buffer + CPU staging)
- Real training metrics and before/after evaluation

Run with: uv run marimo edit notebooks/07_async_rl_e2e.py
"""

# CRITICAL: Set allocator config BEFORE any PyTorch imports (including in subprocesses)
# Set both names for compatibility (old and new PyTorch versions)
import os as _os
_os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


# Cell 1: marimo import
@app.cell
def _():
    import marimo as mo
    return (mo,)


# Cell 2: Environment setup
@app.cell
def _():
    # Environment setup for Monarch subprocesses
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Note: CUDA_VISIBLE_DEVICES is set per-actor in setup()
    # Note: PYTORCH_ALLOC_CONF is set at module level for RDMA

    import sys
    _src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__) if "__file__" in dir() else os.getcwd(), "..", "src"))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    # Set PYTHONPATH for Monarch subprocesses
    _existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{_src_dir}:{_existing}" if _existing else _src_dir
    return


# Cell 3: Title + intro with NB01-03 back-references
@app.cell
def _(mo):
    mo.md(r"""
    # Closing the Loop: Async RL Training

    In **Notebook 04**, we introduced the Zorplex benchmark and saw that Qwen 0.5B
    struggles with compositional tasks -- it gets ~60-70% accuracy when it needs to
    chain multiple tool calls.

    **Now we close the loop**: train the model to get better at these tasks.

    We'll build on patterns from across the series:
    - **`this_host()` from NB02** -- spawning actors on the local machine for interactive development
    - **`try/except` from NB03** -- fault tolerance in generation loops
    - **RDMA weight sync from NB06** -- circular buffer + CPU staging for efficient weight transfer

    The architecture: multiple generators feed trajectories into a replay buffer while
    a trainer continuously samples and updates the policy. We'll measure *before* and
    *after* accuracy to see if training actually helps.
    """)
    return


# Cell 4: Core imports
@app.cell
def _():
    from collections import deque
    import random
    import torch
    import torch.nn.functional as F

    # Model imports
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Zorplex imports
    from zorplex_rl import get_spec, Task
    from zorplex_rl.evaluate import generate_with_tools

    # RL primitives (shared dataclasses)
    from rl_primitives import Trajectory, TrainMetrics

    # RDMA imports (with fallback)
    try:
        from monarch.rdma import RDMABuffer, is_rdma_available
        _rdma_available = is_rdma_available()
    except Exception:
        RDMABuffer = None
        _rdma_available = False

    def rdma_available():
        return _rdma_available
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        F,
        RDMABuffer,
        Task,
        TrainMetrics,
        Trajectory,
        deque,
        generate_with_tools,
        get_spec,
        random,
        rdma_available,
        torch,
    )


# Cell 5: Monarch actor imports
@app.cell
def _():
    from monarch.actor import Actor, endpoint, current_rank
    return Actor, current_rank, endpoint


# Cell 6: Display Trajectory/TrainMetrics fields
@app.cell
def _(Trajectory, TrainMetrics, mo):
    import dataclasses as _dc
    _traj_fields = [(f.name, f.type.__name__ if hasattr(f.type, '__name__') else str(f.type)) for f in _dc.fields(Trajectory)]
    _metrics_fields = [(f.name, f.type.__name__ if hasattr(f.type, '__name__') else str(f.type)) for f in _dc.fields(TrainMetrics)]

    mo.md(f"""
    ## Shared Data Structures

    **Trajectory** -- one rollout from a generator:

    | Field | Type |
    |-------|------|
    {"".join(f"| `{n}` | `{t}` |{chr(10)}" for n, t in _traj_fields)}

    **TrainMetrics** -- returned after each training step:

    | Field | Type |
    |-------|------|
    {"".join(f"| `{n}` | `{t}` |{chr(10)}" for n, t in _metrics_fields)}

    The `model_only_text` field stores the model's generated tokens without injected tool
    results, so the trainer can compute log-probabilities on exactly what the model produced.
    """)
    return


# Cell 7: Service infrastructure (relabeled)
@app.cell
def _(mo):
    mo.md(r"""
    ## Service Infrastructure

    We import a **custom Service abstraction** from `monarch_utils` that manages worker
    replicas with health tracking and round-robin routing. This is a utility we built
    for this notebook series -- the canonical Monarch pattern uses direct actor
    references and slicing, which is what the Service wraps internally.

    (See notebook 05 for the full implementation.)
    """)
    return


# Cell 8: Service imports
@app.cell
def _():
    from monarch_utils.services import Service, register_service
    return Service, register_service


# Cell 9: setup() vs __init__ explanation
@app.cell
def _(mo):
    mo.md(r"""
    ## The `setup()` Pattern

    Monarch actors use a two-phase initialization:

    1. **`__init__`** runs during `spawn()` -- keep it lightweight (store config, set rank)
    2. **`setup()`** is an endpoint called explicitly after spawn -- do heavy work here
       (load models, allocate GPU memory, register RDMA buffers)

    Why not do everything in `__init__`? Two reasons:

    - **`spawn()` is asynchronous** -- it returns immediately, and `__init__` runs in
      the remote process before the first endpoint call. But you don't control *when*,
      and you can't confirm it completed. An explicit `setup()` call lets you sequence
      initialization (e.g., set `CUDA_VISIBLE_DEVICES` and confirm it took effect before
      loading a model).
    - **Coordination** -- you often need to initialize actors in a specific order (set up
      the trainer before generators try to sync weights). Endpoint calls give you that
      sequencing; `__init__` doesn't.
    """)
    return


# Cell 10: ZorplexWorker explanation
@app.cell
def _(mo):
    mo.md(r"""
    ## Actor 1: ZorplexWorker

    Tool execution environments (docker containers, sandboxes, API endpoints) naturally
    form a fleet -- you want many instances running in parallel to keep up with
    generation throughput. That makes them a good fit for a **Service** (from NB05)
    with health tracking and round-robin routing.

    Our ZorplexWorker actors handle Zorplex tasks:
    - `generate_task()` -- creates a new problem
    - `execute_tool()` -- handles LOOKUP calls
    - `check_answer()` -- verifies correctness
    """)
    return


# Cell 11: ZorplexWorker actor
@app.cell
def _(Actor, current_rank, endpoint, get_spec):
    class ZorplexWorker(Actor):
        """Worker actor that handles Zorplex tool execution.

        Managed by a Service for load balancing across replicas.
        """

        def __init__(self, difficulty: str = "easy", seed: int = 42):
            self.rank = current_rank().rank
            self.spec = get_spec("compositional", difficulty=difficulty, seed=seed + self.rank)
            self.calls_served = 0
            print(f"[ZorplexWorker:{self.rank}] Initialized with difficulty={difficulty}")

        @endpoint
        def ping(self) -> bool:
            return True

        @endpoint
        def generate_task(self) -> tuple[str, int]:
            """Generate a new task. Returns (question, correct_answer)."""
            task = self.spec.generate_task()
            return task.question, task.correct_answer

        @endpoint
        def execute_tool(self, tool_name: str, argument: str) -> str:
            """Execute a tool call."""
            from zorplex_rl.task_specs import ToolCall
            tc = ToolCall(tool_name, argument)
            result = self.spec.execute_tool(tc)
            self.calls_served += 1
            return str(result)

        @endpoint
        def get_system_prompt(self) -> str:
            """Get the system prompt with tool hints."""
            return self.spec.get_system_prompt(with_hint=True)

        @endpoint
        def check_answer(self, model_output: str, correct_answer: int) -> tuple[bool, int | None]:
            """Check if model output contains the correct answer."""
            extracted = self.spec.extract_answer(model_output, [])
            is_correct = extracted == correct_answer
            return is_correct, extracted

        @endpoint
        def stats(self) -> dict:
            return {"rank": self.rank, "calls_served": self.calls_served}
    return (ZorplexWorker,)


# Cell 12: ReplayBuffer explanation
@app.cell
def _(mo):
    mo.md(r"""
    ## Actor 2: ReplayBuffer

    A simple actor that stores trajectories. Generators push trajectories in,
    the trainer samples batches out. The `clear()` endpoint lets us reset between
    sync and async runs for a fair comparison.

    **Why a buffer?** Without it, the trainer would train on each trajectory
    immediately after it's generated -- consecutive samples from the same generator
    are highly correlated (same policy, similar tasks). Random sampling from a buffer
    decorrelates the training data, giving better gradient estimates. This is the same
    principle behind experience replay in DQN and other off-policy methods.

    The buffer also enables async RL: generation and training happen independently,
    connected only through the buffer.
    """)
    return


# Cell 13: ReplayBuffer actor (with clear() endpoint)
@app.cell
def _(Actor, Trajectory, deque, endpoint, random):
    class ReplayBuffer(Actor):
        """Stores trajectories for async RL training."""

        def __init__(self, max_size: int = 1000):
            self.buffer: deque[Trajectory] = deque(maxlen=max_size)
            self.total_added = 0
            print(f"[ReplayBuffer] Initialized with max_size={max_size}")

        @endpoint
        def add(self, trajectory: Trajectory) -> None:
            """Add a trajectory to the buffer."""
            self.buffer.append(trajectory)
            self.total_added += 1

        @endpoint
        def sample(self, batch_size: int) -> list[Trajectory]:
            """Sample a batch of trajectories."""
            if len(self.buffer) == 0:
                return []
            n = min(batch_size, len(self.buffer))
            return random.sample(list(self.buffer), n)

        @endpoint
        def size(self) -> int:
            return len(self.buffer)

        @endpoint
        def clear(self) -> int:
            """Clear the buffer. Returns number of items removed."""
            count = len(self.buffer)
            self.buffer.clear()
            return count

        @endpoint
        def stats(self) -> dict:
            if len(self.buffer) == 0:
                return {"size": 0, "total_added": self.total_added, "avg_reward": 0.0}
            rewards = [t.reward for t in self.buffer]
            return {
                "size": len(self.buffer),
                "total_added": self.total_added,
                "avg_reward": sum(rewards) / len(rewards),
                "correct_rate": sum(1 for t in self.buffer if t.is_correct) / len(self.buffer),
            }
    return (ReplayBuffer,)


# Cell 14: TrainerActor explanation (REINFORCE formula, circular buffer, CPU staging)
@app.cell
def _(mo):
    mo.md(r"""
    ## Actor 3: TrainerActor

    The trainer loads the model, receives batches of trajectories, and computes
    REINFORCE updates.

    **REINFORCE**: For each trajectory, we compute per-token log-probabilities on
    the *response tokens only* (using `model_only_text`, not the full text with
    injected tool results). The loss is:

    ```
    loss = -sum(log_prob(token_i)) * (reward - baseline)
    ```

    This weights the entire response by the advantage. Positive advantage reinforces
    the response; negative advantage suppresses it.

    The **baseline** is an exponential moving average (EMA) of past rewards:
    `baseline = 0.9 * baseline + 0.1 * avg_reward`. This reduces gradient variance
    without introducing bias -- the baseline shifts the advantage up or down but
    doesn't change the direction of the gradient in expectation.

    **Approximation note:** In a multi-turn tool-use setting, `model_only_text`
    concatenates the model's generated tokens across turns *without* the injected
    tool results. This means the model sees a different context during training
    (prompt + model text) than during generation (prompt + model text + tool results
    interleaved). The log-probabilities are therefore computed under a slightly
    different distribution than the one that produced the trajectory. This is a
    known approximation -- full-fidelity training would reconstruct the exact
    multi-turn conversation, but that adds significant complexity. For a pedagogical
    notebook, the approximation is acceptable and the gradient signal still points
    in the right direction.

    **Circular buffer with CPU staging** (from NB06): Instead of registering RDMA
    handles directly on GPU parameters (which blocks training during reads), we
    maintain N CPU staging slots. After each training step, weights are copied
    GPU -> CPU (D2H) into the current slot. Generators read from CPU via RDMA,
    then copy to their own GPU (H2D). The data flow:

    ```
    Trainer GPU --D2H--> CPU slot[v % 3] --RDMA--> Generator CPU staging --H2D--> Generator GPU
    ```

    This decouples training from weight distribution.
    """)
    return


# Cell 15: TrainerActor (fixed REINFORCE, circular buffer with CPU staging)
@app.cell
def _(
    Actor,
    AutoModelForCausalLM,
    AutoTokenizer,
    F,
    RDMABuffer,
    TrainMetrics,
    Trajectory,
    current_rank,
    endpoint,
    generate_with_tools,
    get_spec,
    rdma_available,
    torch,
):
    class TrainerActor(Actor):
        """Trains the model on trajectories.

        Uses setup() for heavy initialization (model loading, RDMA registration).
        Implements circular buffer with CPU staging for weight distribution.
        """

        def __init__(
            self,
            model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
            lr: float = 1e-5,
            device: str = "cuda",
            n_buffer_slots: int = 3,
        ):
            # Lightweight init - just store config
            self.model_name = model_name
            self.lr = lr
            self.device_config = device
            self.n_buffer_slots = n_buffer_slots
            self.rank = current_rank().rank
            self._ready = False
            print(f"[Trainer:{self.rank}] Spawned, waiting for setup()...")

        @endpoint
        def setup(self) -> dict:
            """Heavy initialization: load model, create optimizer, set up circular buffer."""
            import os

            if self._ready:
                return {"status": "already_ready"}

            # Trainer always uses GPU 0
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.policy_version = 0
            self.train_steps = 0

            print(f"[Trainer:{self.rank}] Loading model {self.model_name} on GPU 0...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            ).to(self.device)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

            # Store system prompt for log-prob computation in train_step
            self.spec = get_spec("compositional", seed=42)
            self._system_prompt = self.spec.get_system_prompt(with_hint=True)

            # --- Circular buffer with CPU staging ---
            # Compute total bytes for all parameters
            total_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

            # Create N CPU slots for weight staging
            # Note: we use regular (non-pinned) CPU memory here because
            # pin_memory=True triggers dmabuf MR registration in the RDMA
            # subsystem, which can fail on some host configurations.
            self._slots = [
                torch.empty(total_bytes, dtype=torch.uint8)
                for _ in range(self.n_buffer_slots)
            ]

            # Register each slot with RDMA (if available)
            self._slot_handles = []
            if rdma_available() and RDMABuffer is not None:
                try:
                    for slot in self._slots:
                        self._slot_handles.append(RDMABuffer(slot))
                    print(f"[Trainer:{self.rank}] RDMA handles registered for {self.n_buffer_slots} circular buffer slots")
                except Exception as e:
                    print(f"[Trainer:{self.rank}] RDMA registration failed: {e}")
                    self._slot_handles = []

            # Store parameter metadata for scatter on generator side
            self._param_meta = {}
            offset = 0
            for name, p in self.model.named_parameters():
                self._param_meta[name] = (offset, tuple(p.shape), p.dtype)
                offset += p.numel() * p.element_size()

            # Publish initial weights to slot 0
            self._publish_weights()

            self._ready = True
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"[Trainer:{self.rank}] Ready! {param_count:,} params, "
                  f"RDMA={len(self._slot_handles) > 0}, "
                  f"buffer_slots={self.n_buffer_slots}")

            return {
                "status": "ready",
                "params": param_count,
                "rdma": len(self._slot_handles) > 0,
                "buffer_slots": self.n_buffer_slots,
            }

        def _publish_weights(self):
            """Copy GPU params to the current circular buffer slot (D2H)."""
            slot_idx = self.policy_version % self.n_buffer_slots
            slot = self._slots[slot_idx]
            for name, p in self.model.named_parameters():
                off, shape, dtype = self._param_meta[name]
                nbytes = p.numel() * p.element_size()
                slot[off:off + nbytes].copy_(
                    p.data.view(-1).view(torch.uint8).cpu(), non_blocking=True
                )
            torch.cuda.synchronize()  # Ensure D2H complete before RDMA reads

        @endpoint
        def get_weight_handle(self) -> tuple:
            """Get RDMA handle for the latest weight slot.

            Returns (handle_or_None, param_meta, version).
            If RDMA unavailable, handle is None and caller should use get_state_dict().
            """
            if self._slot_handles:
                slot_idx = self.policy_version % self.n_buffer_slots
                return self._slot_handles[slot_idx], self._param_meta, self.policy_version
            return None, self._param_meta, self.policy_version

        @endpoint
        def get_state_dict(self) -> tuple[dict, int]:
            """Fallback: get state dict directly (when RDMA not available)."""
            return self.model.state_dict(), self.policy_version

        @endpoint
        def get_version(self) -> int:
            return self.policy_version

        @endpoint
        def train_step(self, trajectories: list[Trajectory], baseline: float) -> TrainMetrics:
            """Train on a batch of trajectories using REINFORCE.

            For each trajectory:
            1. Reconstruct the prompt using the system prompt + task question
            2. Tokenize prompt to find the boundary
            3. Compute per-token log-probs on response tokens only (model_only_text)
            4. Weight by advantage (reward - baseline)
            """
            if len(trajectories) == 0:
                return TrainMetrics(
                    step=self.train_steps, loss=0.0, batch_size=0,
                    avg_reward=0.0, policy_version=self.policy_version,
                )

            self.model.train()
            self.optimizer.zero_grad()

            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            valid_count = 0

            for traj in trajectories:
                # Reconstruct prompt using system prompt + task question
                messages = [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": traj.task_question},
                ]
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Tokenize prompt to find boundary
                prompt_ids = self.tokenizer(
                    prompt_text, return_tensors="pt", add_special_tokens=False
                )["input_ids"]
                prompt_len = prompt_ids.shape[1]

                # Use model_only_text if available, fall back to response_text
                response = traj.model_only_text if traj.model_only_text else traj.response_text

                # Tokenize prompt + response together
                full_ids = self.tokenizer(
                    prompt_text + response,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=1024,
                )["input_ids"].to(self.device)

                if full_ids.shape[1] <= prompt_len + 1:
                    continue

                with torch.amp.autocast('cuda', enabled=self.device == "cuda"):
                    logits = self.model(full_ids).logits

                # Per-token log-probs on response tokens only
                shift_logits = logits[:, prompt_len - 1:-1, :]
                shift_labels = full_ids[:, prompt_len:]
                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

                advantage = traj.reward - baseline
                total_loss = total_loss + (-token_log_probs.sum() * advantage)
                valid_count += 1

            if valid_count > 0:
                avg_loss = total_loss / valid_count
            else:
                avg_loss = total_loss

            if avg_loss.requires_grad:
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            self.policy_version += 1
            self.train_steps += 1

            # Publish weights to circular buffer (GPU -> CPU)
            self._publish_weights()

            avg_reward = sum(t.reward for t in trajectories) / len(trajectories)

            return TrainMetrics(
                step=self.train_steps,
                loss=avg_loss.item() if torch.is_tensor(avg_loss) else avg_loss,
                batch_size=len(trajectories),
                avg_reward=avg_reward,
                policy_version=self.policy_version,
            )

        @endpoint
        def evaluate_zorplex(self, num_samples: int = 10, seed: int = 42) -> dict:
            """Evaluate current model on compositional Zorplex tasks."""
            self.model.eval()
            spec = get_spec("compositional", seed=seed)
            correct = 0
            total_turns = 0
            total_tools = 0
            for _ in range(num_samples):
                task = spec.generate_task()
                result = generate_with_tools(
                    self.model, self.tokenizer, spec, task,
                    self.device, max_turns=5,
                )
                correct += int(result.is_correct)
                total_turns += len(result.turns)
                total_tools += result.total_tool_calls
            return {
                "accuracy": correct / num_samples,
                "correct": correct,
                "total": num_samples,
                "avg_turns": total_turns / num_samples,
                "avg_tools": total_tools / num_samples,
            }
    return (TrainerActor,)


# Cell 16: GeneratorWorker explanation
@app.cell
def _(mo):
    mo.md(r"""
    ## Actor 4: GeneratorWorker

    Each generator loads its own copy of the model and runs inference independently.
    Weight sync uses the circular buffer pattern:

    1. Generator calls `trainer.get_weight_handle()` to get the RDMA handle + metadata
    2. A single RDMA read pulls the entire contiguous buffer into a pre-allocated
       CPU staging buffer
    3. The generator scatters from CPU staging into GPU model params (H2D)

    This avoids the RDMA API's internal temporary copy (which happens when the
    destination is a GPU tensor) and gives explicit control over timing.

    Fallback path (`sync_weights` using `state_dict`) stays for when RDMA is unavailable.
    """)
    return


# Cell 17: GeneratorWorker (RDMA batching, CPU staging sync)
@app.cell
def _(
    Actor,
    AutoModelForCausalLM,
    AutoTokenizer,
    Task,
    Trajectory,
    current_rank,
    endpoint,
    generate_with_tools,
    get_spec,
    torch,
):
    class GeneratorWorker(Actor):
        """Individual generator worker.

        Uses setup() for heavy initialization (model loading).
        Weight sync uses CPU staging buffer for explicit RDMA -> H2D flow.
        """

        def __init__(
            self,
            model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
            difficulty: str = "easy",
            device: str = "cuda",
        ):
            # Lightweight init - just store config
            self.model_name = model_name
            self.difficulty = difficulty
            self.device_config = device
            self.rank = current_rank().rank
            self._ready = False
            print(f"[GeneratorWorker:{self.rank}] Spawned, waiting for setup()...")

        @endpoint
        def setup(self) -> dict:
            """Heavy initialization: load model, create weight buffer."""
            import os

            if self._ready:
                return {"status": "already_ready"}

            # Generators use GPU 1 + rank (trainer uses GPU 0)
            gpu_id = 1 + self.rank
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.policy_version = 0
            self.generations = 0

            print(f"[GeneratorWorker:{self.rank}] Loading model {self.model_name} on GPU {gpu_id}...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            ).to(self.device)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.spec = get_spec("compositional", difficulty=self.difficulty, seed=42 + self.rank)

            # Pre-allocate CPU staging buffer for weight sync (sized during first sync)
            self._staging_buf = None

            self._ready = True
            print(f"[GeneratorWorker:{self.rank}] Ready on GPU {gpu_id}!")

            return {"status": "ready", "rank": self.rank, "gpu": gpu_id}

        @endpoint
        def get_version(self) -> int:
            return self.policy_version

        @endpoint
        def sync_weights_from_buffer(self, handle, param_meta: dict, version: int) -> bool:
            """Sync weights via RDMA from trainer's circular buffer.

            Flow: Trainer CPU slot --RDMA--> Generator CPU staging --H2D--> Generator GPU
            """
            if version <= self.policy_version:
                return False

            buf_size = handle.size()

            # Allocate staging buffer on first sync (reuse for subsequent syncs)
            if self._staging_buf is None or self._staging_buf.numel() < buf_size:
                self._staging_buf = torch.empty(buf_size, dtype=torch.uint8)

            # RDMA read: trainer CPU memory -> generator CPU memory
            handle.read_into(self._staging_buf[:buf_size]).get()

            # Scatter from CPU staging buffer into GPU model params (H2D)
            for name, p in self.model.named_parameters():
                off, shape, dtype = param_meta[name]
                nbytes = p.numel() * p.element_size()
                src = self._staging_buf[off:off + nbytes].view(dtype).view(shape)
                p.data.copy_(src)  # H2D copy (src is CPU, p.data is on self.device)
            self.policy_version = version
            return True

        @endpoint
        def sync_weights(self, state_dict: dict, version: int) -> bool:
            """Sync weights directly (fallback when RDMA unavailable)."""
            if version <= self.policy_version:
                return False
            self.model.load_state_dict(state_dict)
            self.policy_version = version
            return True

        @endpoint
        def generate(self, question: str, answer: int, max_turns: int = 5) -> Trajectory:
            """Generate a trajectory for the given task."""
            self.model.eval()

            task = Task(question=question, correct_answer=answer, metadata={})

            result = generate_with_tools(
                self.model, self.tokenizer, self.spec, task, self.device,
                max_turns=max_turns, max_tokens_per_turn=150,
            )

            self.generations += 1

            # Build model-only text (generated tokens without injected tool results)
            model_only_text = "".join(t.generated_text for t in result.turns)

            return Trajectory(
                task_question=task.question,
                task_answer=task.correct_answer,
                response_text=result.final_text,
                reward=1.0 if result.is_correct else 0.0,
                is_correct=result.is_correct,
                num_turns=len(result.turns),
                num_tool_calls=result.total_tool_calls,
                generator_id=self.rank,
                policy_version=self.policy_version,
                model_only_text=model_only_text,
            )
    return (GeneratorWorker,)


# Cell 18: Architecture diagram + single-controller back-reference
@app.cell
def _(mo):
    mo.md(r"""
    ## Architecture Overview

    Now we have all our actors defined. Here's how they connect -- this is the
    **single-controller paradigm** from NB01: the notebook process orchestrates
    everything, but actors do the heavy lifting on their own GPUs.

    ```
    ┌─────────────────────┐                    ┌─────────────────────┐
    │  ZorplexService     │                    │  GeneratorWorkers   │
    │  ┌───────────────┐  │     tasks          │  ┌───────────────┐  │
    │  │ ZorplexWorker │  │ ─────────────────> │  │ GenWorker 0   │  │
    │  │ ZorplexWorker │  │                    │  │ GenWorker 1   │  │
    │  └───────────────┘  │                    │  └───────────────┘  │
    └─────────────────────┘                    └──────────┬──────────┘
                                                          │ trajectories
                                                          v
                                               ┌─────────────────────┐
                                               │    ReplayBuffer     │
                                               └──────────┬──────────┘
                                                          │ sample batch
                                                          v
                                               ┌─────────────────────┐
                                               │      Trainer        │
                                               │  (circular buffer)  │──> RDMA weight sync
                                               └─────────────────────┘      to GenWorkers
    ```
    """)
    return


# Cell 19: Sync vs Async explanation + why threading
@app.cell
def _(mo):
    mo.md(r"""
    ## Sync vs Async RL

    **Sync RL** (traditional):
    ```
    |--generate--|--train--|--generate--|--train--|--generate--|--train--|
    ```
    Only ONE thing happens at a time. GPU sits idle during generation,
    generator sits idle during training.

    **Async RL** (what we're building):
    ```
    Gen0:  |████████████████████████████████████████████████████████|
    Gen1:  |████████████████████████████████████████████████████████|
    Train:      |▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓|
    ```
    Everything runs concurrently. More data collected, better GPU utilization.

    **Why threads?** Python's GIL is released during I/O operations and CUDA kernel
    launches. Since our actor calls are remote I/O (`.call_one().get()`) and training
    is GPU-bound, threads achieve real concurrency here. Each generator gets its own
    thread; training runs in the main thread. We cannot use async endpoints in this
    environment, so threading is the right tool.

    **Trade-off: policy staleness.** In async mode, generators produce trajectories
    using an older version of the policy while the trainer has already moved on.
    This is *off-policy* data -- the log-probabilities computed during training
    don't match the policy that generated the trajectory. With REINFORCE this
    introduces some bias. More sophisticated algorithms (PPO, GRPO) address this
    with importance sampling ratios (`pi_new / pi_old`) and clipping, but that's
    beyond our scope here. For a small model with few steps, the staleness is mild.

    We'll run BOTH modes with the **same actors** and compare wall time, throughput,
    and utilization.
    """)
    return


# Cell 20: Configuration sliders
@app.cell
def _(mo):
    num_steps_slider = mo.ui.slider(3, 20, value=5, label="Training steps")
    num_generators_slider = mo.ui.slider(1, 4, value=2, label="Generators")
    batch_size_slider = mo.ui.slider(2, 8, value=4, label="Batch size")

    mo.md(f"""
    ## Configuration

    Adjust parameters for the training run. **Marimo is reactive**: changing a slider
    re-runs all downstream cells that depend on it. This means actors will be
    re-spawned and both training loops will re-execute with the new values.

    {num_steps_slider}

    {num_generators_slider}

    {batch_size_slider}

    **Suggestions:** Start with defaults (5 steps, 2 generators, batch 4) to see the
    full pipeline. Then try increasing generators to 3-4 to see the async throughput
    advantage grow. Increasing training steps gives the model more updates but adds
    wall time. Note that re-spawning actors (loading models onto GPUs) is the most
    expensive part of the setup -- the training loops themselves are relatively fast.
    """)
    return batch_size_slider, num_generators_slider, num_steps_slider


# Cell 21a: Timing dataclasses
@app.cell
def _():
    import threading
    import time
    from dataclasses import dataclass, field

    @dataclass
    class TimingEvent:
        """A single timed event for timeline visualization."""
        actor_id: str
        event_type: str  # "generate", "train", "sync"
        start_time: float
        duration: float

    @dataclass
    class TimingStats:
        """Timing statistics for a training run."""
        mode: str
        num_generators: int
        num_steps: int
        total_generations: int
        wall_time: float
        gen_times: list = field(default_factory=list)
        train_times: list = field(default_factory=list)
        events: list = field(default_factory=list)  # List of TimingEvent
        rdma_syncs: int = 0
        direct_syncs: int = 0

        @property
        def gens_per_second(self) -> float:
            return self.total_generations / self.wall_time if self.wall_time > 0 else 0

        @property
        def steps_per_second(self) -> float:
            return self.num_steps / self.wall_time if self.wall_time > 0 else 0
    return (
        TimingEvent,
        TimingStats,
        field,
        threading,
        time,
    )


# Cell 21b: Configuration constants + setup_actors
@app.cell
def _(
    GeneratorWorker,
    ReplayBuffer,
    Service,
    TrainerActor,
    ZorplexWorker,
    batch_size_slider,
    num_generators_slider,
    num_steps_slider,
    register_service,
):
    from monarch.actor import this_host

    NUM_STEPS = num_steps_slider.value
    NUM_GENERATORS = num_generators_slider.value
    NUM_ZORPLEX = 2
    BATCH_SIZE = batch_size_slider.value

    def setup_actors():
        """Spawn and initialize all actors. Returns them for reuse."""
        host = this_host()

        # 1. ZorplexWorkers -- Service spawns workers internally
        zorplex_worker_procs = host.spawn_procs(per_host={"procs": NUM_ZORPLEX})
        zorplex_svc_procs = host.spawn_procs(per_host={"procs": 1})
        zorplex_svc = zorplex_svc_procs.spawn("zorplex_svc", Service,
            service_name="zorplex", worker_class=ZorplexWorker,
            procs=zorplex_worker_procs, procs_per_replica=1,
            difficulty="easy")

        # 2. GeneratorWorkers -- spawned directly (no Service wrapper for individual access)
        gen_worker_procs = host.spawn_procs(per_host={"procs": NUM_GENERATORS})
        gen_svc_procs = host.spawn_procs(per_host={"procs": 1})
        gen_svc = gen_svc_procs.spawn("gen_svc", Service,
            service_name="generators", worker_class=GeneratorWorker,
            procs=gen_worker_procs, procs_per_replica=1)

        # 3. ReplayBuffer
        buffer_procs = host.spawn_procs(per_host={"procs": 1})
        buffer = buffer_procs.spawn("buffer", ReplayBuffer, max_size=500)

        # 4. Trainer
        trainer_procs = host.spawn_procs(per_host={"procs": 1})
        trainer = trainer_procs.spawn("trainer", TrainerActor)

        # Initialize actors that need setup
        zorplex_svc.ping.call_one().get()

        print("[SETUP] Setting up generator workers...")
        gen_worker_list = gen_svc.get_all_replicas.call_one().get()
        for w in gen_worker_list:
            w.setup.call_one().get()

        gen_svc.ping.call_one().get()
        buffer.stats.call_one().get()

        print("[SETUP] Setting up trainer...")
        trainer.setup.call_one().get()

        register_service("zorplex", zorplex_svc)
        register_service("generators", gen_svc)

        print(f"[SETUP] All actors ready! {NUM_GENERATORS} generators, {NUM_ZORPLEX} zorplex workers")

        return {
            "trainer": trainer,
            "buffer": buffer,
            "gen_worker_list": gen_worker_list,
            "gen_svc": gen_svc,
            "zorplex_svc": zorplex_svc,
        }

    actors = setup_actors()
    return (
        BATCH_SIZE,
        NUM_GENERATORS,
        NUM_STEPS,
        NUM_ZORPLEX,
        actors,
        setup_actors,
        this_host,
    )


# Cell 22: "Before Training: Zorplex Baseline"
@app.cell
def _(mo):
    mo.md(r"""
    ## Before Training: Zorplex Baseline

    Let's evaluate the model *before* any training to establish a baseline.
    We run 10 compositional Zorplex tasks and record accuracy, average turns,
    and tool usage. This gives us a concrete "before" snapshot to compare against.
    """)
    return


# Cell 23: Pre-training evaluation
@app.cell
def _(actors, mo):
    print("Evaluating pre-training baseline...")
    pre_eval = actors["trainer"].evaluate_zorplex.call_one(num_samples=10, seed=42).get()

    mo.md(f"""
    ### Pre-Training Results

    | Metric | Value |
    |--------|-------|
    | Accuracy | {pre_eval['accuracy']:.0%} ({pre_eval['correct']}/{pre_eval['total']}) |
    | Avg turns | {pre_eval['avg_turns']:.1f} |
    | Avg tool calls | {pre_eval['avg_tools']:.1f} |

    This is our starting point. Let's see if training improves it.
    """)
    return (pre_eval,)


# Cell 24: run_sync_loop (uses ALL generators sequentially)
@app.cell
def _(BATCH_SIZE, NUM_STEPS, TimingEvent, TimingStats, actors, time):
    def run_sync_loop() -> TimingStats:
        """
        SYNC MODE: Rotate through ALL generators sequentially.
        Pattern: generate -> train -> generate -> train ...

        Uses all generators in round-robin so the comparison with async
        is fair -- same number of generators, just used sequentially.
        """
        print("\n" + "=" * 60)
        print("SYNC MODE: Sequential Generate -> Train")
        print("=" * 60)

        trainer = actors["trainer"]
        buffer = actors["buffer"]
        gen_worker_list = actors["gen_worker_list"]
        zorplex_svc = actors["zorplex_svc"]

        stats = TimingStats(
            mode="SYNC",
            num_generators=len(gen_worker_list),
            num_steps=NUM_STEPS,
            total_generations=0,
            wall_time=0,
        )

        baseline = 0.5
        t0 = time.perf_counter()

        for step in range(NUM_STEPS):
            # Rotate through ALL generators (fair comparison with async)
            gen_worker = gen_worker_list[step % len(gen_worker_list)]

            # Generate ONE trajectory (blocking)
            gen_start = time.perf_counter()
            zorplex_worker, _ = zorplex_svc.get_replica_with_idx.call_one().get()
            question, answer = zorplex_worker.generate_task.call_one().get()
            traj = gen_worker.generate.call_one(question, answer).get()
            buffer.add.call_one(traj).get()
            gen_time = time.perf_counter() - gen_start
            stats.gen_times.append(gen_time)
            stats.total_generations += 1
            stats.events.append(TimingEvent(
                actor_id=f"Gen{step % len(gen_worker_list)}",
                event_type="generate",
                start_time=gen_start - t0,
                duration=gen_time,
            ))

            # Train on buffer (blocking)
            train_start = time.perf_counter()
            batch = buffer.sample.call_one(BATCH_SIZE).get()
            if batch:
                metrics = trainer.train_step.call_one(batch, baseline).get()
                baseline = 0.9 * baseline + 0.1 * metrics.avg_reward
            train_time = time.perf_counter() - train_start
            stats.train_times.append(train_time)
            stats.events.append(TimingEvent(
                actor_id="Train",
                event_type="train",
                start_time=train_start - t0,
                duration=train_time,
            ))

            status = "correct" if traj.is_correct else "wrong"
            print(f"[SYNC {step + 1:2d}] {status} gen={gen_time * 1000:.0f}ms train={train_time * 1000:.0f}ms")

        stats.wall_time = time.perf_counter() - t0
        return stats
    return (run_sync_loop,)


# Cell 25: run_async_loop (1 thread per generator, separate sync thread)
@app.cell
def _(BATCH_SIZE, NUM_STEPS, TimingEvent, TimingStats, actors, threading, time):
    def run_async_loop() -> TimingStats:
        """
        ASYNC MODE: All generators running concurrently with trainer.
        - 1 thread per generator (each runs its own generation loop)
        - 1 separate weight sync thread
        - Training in main thread

        Uses try/except pattern from NB03 for fault tolerance in generation loops.
        """
        print("\n" + "=" * 60)
        print(f"ASYNC MODE: {len(actors['gen_worker_list'])} Generators + 1 Trainer (Concurrent)")
        print("=" * 60)

        trainer = actors["trainer"]
        buffer = actors["buffer"]
        gen_worker_list = actors["gen_worker_list"]
        zorplex_svc = actors["zorplex_svc"]

        stats = TimingStats(
            mode="ASYNC",
            num_generators=len(gen_worker_list),
            num_steps=NUM_STEPS,
            total_generations=0,
            wall_time=0,
        )

        lock = threading.Lock()
        stop_flag = threading.Event()
        t0 = time.perf_counter()

        def generation_loop(gen_idx, gen_worker):
            """Each generator gets its own thread."""
            while not stop_flag.is_set():
                gen_start = time.perf_counter()

                zorplex_worker, _ = zorplex_svc.get_replica_with_idx.call_one().get()
                question, answer = zorplex_worker.generate_task.call_one().get()

                try:
                    traj = gen_worker.generate.call_one(question, answer).get()
                    buffer.add.call_one(traj).get()

                    gen_time = time.perf_counter() - gen_start
                    with lock:
                        stats.gen_times.append(gen_time)
                        stats.total_generations += 1
                        count = stats.total_generations
                        stats.events.append(TimingEvent(
                            actor_id=f"Gen{gen_idx}",
                            event_type="generate",
                            start_time=gen_start - t0,
                            duration=gen_time,
                        ))

                    status = "correct" if traj.is_correct else "wrong"
                    print(f"[GEN{gen_idx} #{count:2d}] {status} gen={gen_time * 1000:.0f}ms")

                except Exception as e:
                    # try/except pattern from NB03 -- log and continue
                    print(f"[GEN{gen_idx}] Error: {e}, retrying...")
                    continue

        def weight_sync_loop():
            """Periodically sync weights to all generators."""
            while not stop_flag.is_set():
                time.sleep(0.5)  # Sync every 500ms
                if stop_flag.is_set():
                    break

                sync_start = time.perf_counter()
                try:
                    handle, param_meta, version = trainer.get_weight_handle.call_one().get()
                    for w in gen_worker_list:
                        try:
                            if handle is not None:
                                synced = w.sync_weights_from_buffer.call_one(handle, param_meta, version).get()
                                if synced:
                                    with lock:
                                        stats.rdma_syncs += 1
                            else:
                                state_dict, ver = trainer.get_state_dict.call_one().get()
                                synced = w.sync_weights.call_one(state_dict, ver).get()
                                if synced:
                                    with lock:
                                        stats.direct_syncs += 1
                        except Exception:
                            pass

                    sync_time = time.perf_counter() - sync_start
                    with lock:
                        stats.events.append(TimingEvent(
                            actor_id="Sync",
                            event_type="sync",
                            start_time=sync_start - t0,
                            duration=sync_time,
                        ))
                except Exception:
                    pass

        # Start 1 thread per generator
        gen_threads = []
        for idx, worker in enumerate(gen_worker_list):
            t = threading.Thread(target=generation_loop, args=(idx, worker), daemon=True)
            t.start()
            gen_threads.append(t)

        # Start weight sync thread
        sync_thread = threading.Thread(target=weight_sync_loop, daemon=True)
        sync_thread.start()

        # Training in main thread
        train_steps_done = 0
        baseline = 0.5

        while train_steps_done < NUM_STEPS:
            # Wait for enough samples
            while True:
                size = buffer.size.call_one().get()
                if size >= BATCH_SIZE:
                    break
                time.sleep(0.02)

            train_start = time.perf_counter()
            batch = buffer.sample.call_one(BATCH_SIZE).get()
            if batch:
                metrics = trainer.train_step.call_one(batch, baseline).get()
                baseline = 0.9 * baseline + 0.1 * metrics.avg_reward

            train_time = time.perf_counter() - train_start
            with lock:
                stats.train_times.append(train_time)
                stats.events.append(TimingEvent(
                    actor_id="Train",
                    event_type="train",
                    start_time=train_start - t0,
                    duration=train_time,
                ))
            train_steps_done += 1

            print(f"[TRAIN {train_steps_done:2d}] time={train_time * 1000:.0f}ms buffer={size}")

        stop_flag.set()

        for t in gen_threads:
            t.join(timeout=2.0)
        sync_thread.join(timeout=2.0)

        stats.wall_time = time.perf_counter() - t0
        return stats
    return (run_async_loop,)


# Cell 26: Clear buffer between runs, execute sync loop
@app.cell
def _(actors, run_sync_loop):
    # Clear buffer to start fresh
    _cleared = actors["buffer"].clear.call_one().get()
    print(f"Buffer cleared ({_cleared} items removed)")

    sync_stats = run_sync_loop()
    print(f"\nSync complete: {sync_stats.wall_time:.2f}s, "
          f"{sync_stats.total_generations} generations, "
          f"{sync_stats.gens_per_second:.2f} gens/s")
    return (sync_stats,)


# Cell 27: Execute async loop
@app.cell
def _(actors, run_async_loop):
    # Clear buffer between runs for fair comparison
    _cleared = actors["buffer"].clear.call_one().get()
    print(f"Buffer cleared ({_cleared} items removed)")

    async_stats = run_async_loop()
    print(f"\nAsync complete: {async_stats.wall_time:.2f}s, "
          f"{async_stats.total_generations} generations, "
          f"{async_stats.gens_per_second:.2f} gens/s")
    return (async_stats,)


# Cell 28: Comparison table + markdown
@app.cell
def _(async_stats, mo, sync_stats):
    def _build_comparison(sync_s, async_s) -> str:
        speedup = sync_s.wall_time / async_s.wall_time if async_s.wall_time > 0 else 0
        gen_ratio = async_s.gens_per_second / sync_s.gens_per_second if sync_s.gens_per_second > 0 else 0

        avg_sync_gen = sum(sync_s.gen_times) / len(sync_s.gen_times) * 1000 if sync_s.gen_times else 0
        avg_async_gen = sum(async_s.gen_times) / len(async_s.gen_times) * 1000 if async_s.gen_times else 0
        avg_sync_train = sum(sync_s.train_times) / len(sync_s.train_times) * 1000 if sync_s.train_times else 0
        avg_async_train = sum(async_s.train_times) / len(async_s.train_times) * 1000 if async_s.train_times else 0

        sync_type = f"{async_s.rdma_syncs} RDMA" if async_s.rdma_syncs > 0 else f"{async_s.direct_syncs} direct"

        return f"""
## Sync vs Async Comparison

| Metric | SYNC | ASYNC | Ratio |
|--------|------|-------|-------|
| Wall time | {sync_s.wall_time:.2f}s | {async_s.wall_time:.2f}s | **{speedup:.2f}x** speedup |
| Generations | {sync_s.total_generations} | {async_s.total_generations} | {async_s.total_generations / max(sync_s.total_generations, 1):.1f}x |
| Gens/second | {sync_s.gens_per_second:.2f} | {async_s.gens_per_second:.2f} | **{gen_ratio:.1f}x** throughput |
| Avg gen time | {avg_sync_gen:.0f}ms | {avg_async_gen:.0f}ms | |
| Avg train time | {avg_sync_train:.0f}ms | {avg_async_train:.0f}ms | |
| Weight syncs | -- | {sync_type} | |

### Key Observations

- **Data throughput**: Async collected **{gen_ratio:.1f}x** more trajectories per second.
  More data means better gradient estimates.
- **GPU utilization**: In sync mode, the trainer GPU sits idle during generation and
  vice versa. Async keeps both busy.
- **Generators ran in parallel**: {async_s.num_generators} generators each had their own
  thread, producing data independently.
- The trainer consumed from the replay buffer continuously, never waiting for a specific
  generator to finish.

In production with more generators, the throughput advantage grows further.
"""

    comparison_md = _build_comparison(sync_stats, async_stats)
    mo.md(comparison_md)
    return (comparison_md,)


# Cell 29: Timeline visualization (matplotlib Gantt charts)
@app.cell
def _(async_stats, mo, sync_stats):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    def _plot_timeline(stats, ax, title):
        """Plot a Gantt chart of timing events."""
        color_map = {
            "generate": "#4CAF50",  # green
            "train": "#E91E63",     # pink
            "sync": "#9C27B0",      # purple
        }

        # Collect unique actor IDs and assign y positions
        actor_ids = []
        for ev in stats.events:
            if ev.actor_id not in actor_ids:
                actor_ids.append(ev.actor_id)

        # Sort: Gen0, Gen1, ..., Train, Sync
        gen_ids = sorted([a for a in actor_ids if a.startswith("Gen")])
        other_ids = [a for a in ["Train", "Sync"] if a in actor_ids]
        actor_ids = gen_ids + other_ids

        y_map = {aid: i for i, aid in enumerate(actor_ids)}

        for ev in stats.events:
            if ev.actor_id in y_map:
                y = y_map[ev.actor_id]
                color = color_map.get(ev.event_type, "#999999")
                ax.barh(y, ev.duration, left=ev.start_time, height=0.6,
                        color=color, alpha=0.8, edgecolor="white", linewidth=0.5)

        ax.set_yticks(range(len(actor_ids)))
        ax.set_yticklabels(actor_ids)
        ax.set_xlabel("Wall time (seconds)")
        ax.set_title(title)
        ax.invert_yaxis()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    _plot_timeline(sync_stats, ax1, f"SYNC ({sync_stats.wall_time:.1f}s)")
    _plot_timeline(async_stats, ax2, f"ASYNC ({async_stats.wall_time:.1f}s)")

    # Legend
    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="Generate"),
        mpatches.Patch(color="#E91E63", label="Train"),
        mpatches.Patch(color="#9C27B0", label="Weight Sync"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    mo.md("""### Timeline Visualization

The Gantt charts below show what each actor was doing over time. In sync mode,
bars are strictly sequential. In async mode, generators and trainer overlap --
that overlap is where the throughput gain comes from.
""")
    return


# Cell 30: "After Training: Did It Improve?"
@app.cell
def _(mo):
    mo.md(r"""
    ## After Training: Did It Improve?

    We've now run both sync and async training loops. The model has been updated
    through multiple training steps. Let's evaluate the same set of compositional
    tasks to see if accuracy changed.

    Note: We're using a small model (0.5B) with few training steps, so dramatic
    improvement isn't guaranteed. The point is the *infrastructure* -- showing that
    the full loop works end to end.
    """)
    return


# Cell 31: Post-training evaluation
@app.cell
def _(actors):
    print("Evaluating post-training performance...")
    post_eval = actors["trainer"].evaluate_zorplex.call_one(num_samples=10, seed=42).get()
    print(f"Post-training accuracy: {post_eval['accuracy']:.0%}")
    return (post_eval,)


# Cell 32: Before/after comparison display
@app.cell
def _(mo, post_eval, pre_eval):
    _acc_delta = post_eval["accuracy"] - pre_eval["accuracy"]
    _direction = "improved" if _acc_delta > 0 else ("unchanged" if _acc_delta == 0 else "decreased")

    mo.md(f"""
    ### Before vs After Training

    | Metric | Before | After | Delta |
    |--------|--------|-------|-------|
    | Accuracy | {pre_eval['accuracy']:.0%} | {post_eval['accuracy']:.0%} | {_acc_delta:+.0%} |
    | Avg turns | {pre_eval['avg_turns']:.1f} | {post_eval['avg_turns']:.1f} | {post_eval['avg_turns'] - pre_eval['avg_turns']:+.1f} |
    | Avg tool calls | {pre_eval['avg_tools']:.1f} | {post_eval['avg_tools']:.1f} | {post_eval['avg_tools'] - pre_eval['avg_tools']:+.1f} |

    Accuracy {_direction} by {abs(_acc_delta):.0%}.

    With a 0.5B model and only a few training steps, large gains are unlikely.
    The key result is that the full pipeline works: generation, buffering, training,
    weight sync, and evaluation all compose correctly through Monarch actors.
    """)
    return


# Cell 33: What's happening under the hood (updated)
@app.cell
def _(mo):
    mo.md(r"""
    ## What's Happening Under the Hood

    When you run the training loop, here's what each layer does:

    **Actor isolation**: Each actor (trainer, generators, buffer, zorplex workers)
    runs in its own process with its own GPU assignment. `CUDA_VISIBLE_DEVICES` is
    set in `setup()`, not at spawn time -- the `procs` dimension in `spawn_procs`
    is just a dimension name, not a GPU assignment.

    **Weight sync data flow** (circular buffer + CPU staging from NB06):
    ```
    Trainer GPU  --D2H-->  CPU slot[v % 3]  --RDMA-->  Generator CPU staging  --H2D-->  Generator GPU
    ```
    - Trainer publishes weights to a circular buffer after each train step
    - Generators pull from the buffer via RDMA into a pre-allocated staging buffer
    - Explicit H2D copy scatters into GPU model parameters
    - The circular buffer has 3 slots, so training never blocks on reads

    **Async concurrency** (via threads):
    - 1 thread per generator, each running its own generation loop
    - 1 weight sync thread, periodically pulling from trainer and pushing to generators
    - Training in the main thread
    - `threading.Event` coordinates shutdown when training completes
    - GIL is released during I/O (actor calls) and CUDA (GPU compute), so threads
      achieve real concurrency

    **Fault tolerance** (from NB03):
    - Generation loops wrap `generate.call_one().get()` in `try/except`
    - On failure, the generator logs and retries instead of crashing the loop
    """)
    return


# Cell 34: Scaling up
@app.cell
def _(mo):
    mo.md(r"""
    ## Scaling Up

    What we built here scales naturally with Monarch:

    | Scale | What Changes |
    |-------|--------------|
    | More generators | Increase `num_generators` slider -- spawns larger ActorMesh |
    | More zorplex workers | Increase `NUM_ZORPLEX` -- parallel task generation |
    | Multi-node | Use `SlurmJob` instead of `this_host()` |
    | Better algorithms | Swap REINFORCE for PPO/GRPO |
    | More services | Add reward models, search APIs as actors |

    **The patterns stay the same:**
    - Actors for isolation and GPU assignment
    - Endpoints for communication (`.call_one().get()`)
    - RDMA + circular buffer for efficient weight transfer
    - Version tracking for consistency across actors

    This is the foundation for production systems like Forge GRPO, which uses the
    same Monarch actor patterns at much larger scale.
    """)
    return


# Cell 35: Recap (NB01-07 summary, forward-looking conclusion)
@app.cell
def _(mo):
    mo.md(r"""
    ## Recap: The Full Journey

    We've come a long way in this notebook series:

    | Notebook | What We Learned |
    |----------|-----------------|
    | 01 | Monarch's history and the single-controller paradigm |
    | 02 | Interactive development with `this_host()` |
    | 03 | Fault tolerance with `try/except` on actor calls |
    | 04 | Zorplex benchmark -- where Qwen 0.5B struggles |
    | 05 | Services for managing worker pools with health tracking |
    | 06 | RDMA weight sync, circular buffers, CPU staging |
    | **07** | **Closing the loop: async RL training end to end** |

    **Key takeaways from this notebook:**

    - Monarch makes distributed RL feel like local Python -- actors, endpoints,
      and slicing compose naturally into a full training system
    - Async RL collects more data per unit wall time by running generators
      and trainer concurrently
    - The circular buffer + CPU staging pattern from NB06 decouples training
      from weight distribution
    - Before/after evaluation closes the loop: we can measure whether training
      actually improves the model

    **Where to go next:** Forge GRPO implements these same patterns at production
    scale -- multiple nodes, larger models, PPO/GRPO instead of REINFORCE, and
    proper reward modeling. The Monarch primitives you've learned here are the
    building blocks for all of it.
    """)
    return


if __name__ == "__main__":
    app.run()
