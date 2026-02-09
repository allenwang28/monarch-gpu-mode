import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # Shared imports for the notebook
    import time
    import torch
    from monarch.actor import Actor, endpoint, this_host, current_rank
    from monarch.rdma import RDMABuffer, RDMAAction, is_rdma_available
    return (
        Actor,
        RDMAAction,
        RDMABuffer,
        current_rank,
        endpoint,
        is_rdma_available,
        this_host,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # RDMA Deep Dive

    Notebook 07 covered weight synchronization end-to-end: why RDMA matters for async RL,
    the magic pointer pattern, circular buffers, and re-sharding. We used `RDMABuffer` and
    `read_into()` as black boxes — call the API, weights appear.

    This notebook goes deeper into how RDMA actually works. We'll walk through the ibverbs
    primitives that power every transfer, understand the costs that separate a fast
    implementation from a slow one, and see the concrete patterns for managing RDMA buffers
    in production.

    **The central question: where do the milliseconds go?**

    1. **ibverbs Internals** — Queue Pairs, Memory Registration, Completion Queues
    2. **RDMA Buffer Patterns** — Three approaches to managing registration and transfers

    *We focus on **ibverbs** because that's what Monarch's RDMA subsystem supports today
    (InfiniBand and RoCE via Mellanox/NVIDIA ConnectX NICs). EFA (AWS Elastic Fabric Adapter)
    is a relevant transport but not yet supported — it's actively being worked on.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. ibverbs Internals

    RDMA (Remote Direct Memory Access) lets one machine read/write another machine's memory
    directly, bypassing the kernel and CPU on both sides.

    ### The ibverbs Stack

    ```
    ┌─────────────────────────────────────────────────────────┐
    │  Application (PyTorch, Monarch, etc.)                   │
    ├─────────────────────────────────────────────────────────┤
    │  libibverbs  (userspace RDMA API)                       │
    ├─────────────────────────────────────────────────────────┤
    │  Provider driver (mlx5, efa, rxe, etc.)                 │
    ├─────────────────────────────────────────────────────────┤
    │  Hardware (InfiniBand NIC, RoCE NIC, etc.)              │
    └─────────────────────────────────────────────────────────┘
    ```

    We focus on **InfiniBand** and **RoCE** (RDMA over Converged Ethernet) — the two
    transports Monarch supports today via the ibverbs API.

    ### Key RDMA Operations

    | Operation | Description |
    |-----------|-------------|
    | `RDMA_WRITE` | Write to remote memory (one-sided) |
    | `RDMA_READ` | Read from remote memory (one-sided) |
    | `SEND/RECV` | Two-sided messaging (like TCP) |

    The magic is in `RDMA_WRITE` and `RDMA_READ` - they're **one-sided**:
    - Remote CPU is not involved
    - Remote application doesn't need to call anything
    - NIC handles everything in hardware

    ### Memory Registration

    Before RDMA, memory must be **registered** with the NIC:

    ```python
    # Conceptually (actual ibverbs API is in C)
    mr = rdma_register_memory(buffer, size)
    # Returns:
    #   - lkey: local access key (for local operations)
    #   - rkey: remote access key (share with remote peer)
    #   - addr: physical/virtual address
    ```

    The `(addr, rkey)` pair is a **remote-accessible pointer**. Share it with a peer,
    and they can read/write your memory directly.

    ### Queue Pair Setup

    Before any RDMA operations, you need to establish a **Queue Pair (QP)** between
    sender and receiver. This is a one-time connection setup:

    ```
    ┌─────────────┐                           ┌─────────────┐
    │   Sender    │                           │  Receiver   │
    │             │                           │             │
    │  Create QP  │ ─── exchange QP info ───► │  Create QP  │
    │  (qp_num,   │ ◄── (qp_num, lid, gid) ── │             │
    │   lid, gid) │                           │             │
    │             │                           │             │
    │  Move QP to │                           │  Move QP to │
    │  RTR → RTS  │                           │  RTR → RTS  │
    │             │                           │             │
    │  Now ready  │ ═══ RDMA operations ════► │  Now ready  │
    │  for RDMA!  │                           │  for RDMA!  │
    └─────────────┘                           └─────────────┘
    ```

    This is where **Monarch actors** shine. Because you can spawn arbitrary actors,
    you can create **RDMA Manager actors** that:
    - Initialize QPs on their respective hosts
    - Exchange QP info via actor messages
    - Manage the connection lifecycle

    ```python
    # Monarch pattern: RDMA managers as actors
    class RDMAManager(Actor):
        def __init__(self):
            self.qp = create_queue_pair()
            self.qp_info = get_qp_info(self.qp)  # (qp_num, lid, gid)

        @endpoint
        def get_qp_info(self) -> QpInfo:
            return self.qp_info

        @endpoint
        def connect(self, remote_qp_info: QpInfo):
            # Transition QP: INIT → RTR → RTS
            connect_qp(self.qp, remote_qp_info)

    # Setup: exchange QP info via actor messages, then RDMA is ready
    trainer_info = trainer_rdma.get_qp_info.call_one().get()
    generator_rdma.connect.call_one(trainer_info).get()
    ```

    The actor abstraction makes RDMA connection management natural and composable.

    ### Completion Queues (CQs)

    How does the initiator know an operation finished? Every QP is associated with a
    **Completion Queue (CQ)**. When the NIC finishes an
    RDMA operation, it posts a **completion event** to the CQ:

    ```
    App: post RDMA_READ to Send Queue
              ↓
    NIC: executes the read (bypasses remote CPU)
              ↓
    NIC: posts completion to CQ
              ↓
    App: poll CQ → "done, 4096 bytes transferred"
    ```

    Monarch's Rust layer (`RdmaManagerActor`) handles CQ polling internally. When you
    call `.get()` on an RDMA future, you're ultimately waiting for a CQ completion event.
    This is why RDMA is "one-sided" but not "zero-sided" - the *initiator* still needs
    to know when the transfer is done.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Monarch Using Monarch: RdmaController

    Here's the cool part: **Monarch uses itself** to manage RDMA infrastructure. Looking at
    the actual Python code in `monarch/_src/rdma/rdma.py`:

    ```python
    # From Monarch's RDMA implementation
    from monarch._src.actor.proc_mesh import get_or_spawn_controller

    class RdmaController(Actor):
        '''Singleton controller that coordinates RDMA initialization.'''

        def __init__(self):
            # Track which proc meshes have RDMA initialized
            self._manager_futures: dict[ProcMesh, Future[RdmaManager]] = {}

        @endpoint
        async def init_rdma_on_mesh(self, proc_mesh: ProcMesh) -> None:
            '''Lazily initialize RDMA on a proc mesh.'''
            if proc_mesh not in self._manager_futures:
                self._manager_futures[proc_mesh] = Future(
                    coro=RdmaManager.create(proc_mesh)
                )
            await self._manager_futures[proc_mesh]

    # Cached initialization - only runs once per process
    @functools.cache
    def _ensure_init_rdma_manager():
        async def task():
            controller = await get_or_spawn_controller("rdma_controller", RdmaController)
            await controller.init_rdma_on_mesh.call_one(current_proc_mesh())
        return spawn_task(task())
    ```

    This is **Monarch building Monarch** - the RDMA subsystem uses the same patterns:

    - `get_or_spawn_controller("rdma_controller", RdmaController)` ensures one global controller
    - The controller lazily initializes RDMA managers per proc mesh
    - `@functools.cache` ensures we only bootstrap once per process
    - Under the hood, the actual RDMA operations are in Rust (`RdmaManagerActor`)

    It's actors all the way down.

    Now that we understand the primitives - QPs, MRs, CQs, and how Monarch wraps them -
    the next question is: **what's the cost of getting them wrong?** Memory registration
    turns out to be the biggest hidden tax.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. RDMA Buffer Patterns

    Memory registration is the hidden cost of RDMA. Before the NIC can read or write a
    buffer, that buffer must be **registered** — the OS pins its physical pages and creates
    DMA mappings so the NIC can access them directly. This can take milliseconds for large
    buffers.

    The question isn't *whether* to pay this cost — it's *when and how often*. We'll
    look at three patterns:

    | Pattern | Registration | Transfer | Trade-off |
    |---------|-------------|----------|-----------|
    | **Naive** | Every call | Per-buffer | Baseline — pays MR cost every step |
    | **Contiguous** | Once at init | One bulk read | Fastest, but requires copying to a contiguous region |
    | **Scattered + RDMAAction** | Once at init | Batched plan | Practical — works with non-contiguous layouts |

    The benchmark at the end shows the performance difference.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Pattern 1: Naive

    Create a new `RDMABuffer` on every transfer — pays memory registration cost each time:
    """)
    return


@app.cell
def _(Actor, RDMABuffer, current_rank, endpoint, time, torch):
    class NaiveSender(Actor):
        """Creates new RDMABuffer handles every transfer. Expensive!"""

        def __init__(self, layer_sizes: list):
            self.layer_sizes = layer_sizes
            self.layers = [torch.zeros(size, dtype=torch.float32) for size in layer_sizes]
            for i, layer in enumerate(self.layers):
                layer.fill_(float(i + 1))

        @endpoint
        def get_fresh_handles(self) -> list:
            handles = []
            for size, layer in zip(self.layer_sizes, self.layers):
                byte_view = layer.view(torch.uint8).flatten()
                handles.append((size, RDMABuffer(byte_view)))
            return handles


    class NaiveReceiver(Actor):
        """Receives from naive sender - pays MR cost every step."""

        def __init__(self, layer_sizes: list):
            self.layer_sizes = layer_sizes
            self.layers = [torch.zeros(size, dtype=torch.float32) for size in layer_sizes]
            self.rank = current_rank().rank

        @endpoint
        def warmup(self, sender: NaiveSender):
            """Force RDMA subsystem init + QP establishment (not timed)."""
            handles = sender.get_fresh_handles.call_one().get()
            for i, (size, handle) in enumerate(handles):
                byte_view = self.layers[i].view(torch.uint8).flatten()
                handle.read_into(byte_view).get()

        @endpoint
        def receive_step(self, sender: NaiveSender) -> dict:
            start = time.perf_counter()
            handles = sender.get_fresh_handles.call_one().get()
            for i, (size, handle) in enumerate(handles):
                byte_view = self.layers[i].view(torch.uint8).flatten()
                handle.read_into(byte_view).get()
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"elapsed_ms": elapsed_ms}

    print("NaiveSender: Re-registers all parameters on every call")
    return NaiveReceiver, NaiveSender


@app.cell
def _(mo):
    mo.md(r"""
    ### Pattern 2: Contiguous Buffer

    Pack all layers into one buffer, register once — one MR, one transfer:
    """)
    return


@app.cell
def _(Actor, RDMABuffer, current_rank, endpoint, time, torch):
    class ContiguousSender(Actor):
        """One buffer, one MR, registered at startup."""

        def __init__(self, layer_sizes: list):
            self.layer_sizes = layer_sizes
            total_size = sum(layer_sizes)

            # One contiguous buffer
            self.buffer = torch.zeros(total_size, dtype=torch.float32)
            offset = 0
            for i, size in enumerate(layer_sizes):
                self.buffer[offset : offset + size].fill_(float(i + 1))
                offset += size

            # Register ONCE at startup
            byte_view = self.buffer.view(torch.uint8).flatten()
            self.handle = RDMABuffer(byte_view)

        @endpoint
        def get_handle(self) -> tuple:
            return (len(self.buffer), self.handle)  # Same handle every time!


    class ContiguousReceiver(Actor):
        """Receives from contiguous sender - one big read."""

        def __init__(self, total_size: int):
            self.buffer = torch.zeros(total_size, dtype=torch.float32)
            self.rank = current_rank().rank

        @endpoint
        def warmup(self, sender: ContiguousSender):
            """Force QP establishment (not timed)."""
            size, handle = sender.get_handle.call_one().get()
            byte_view = self.buffer.view(torch.uint8).flatten()
            handle.read_into(byte_view).get()

        @endpoint
        def receive_step(self, sender: ContiguousSender) -> dict:
            start = time.perf_counter()
            size, handle = sender.get_handle.call_one().get()
            byte_view = self.buffer.view(torch.uint8).flatten()
            handle.read_into(byte_view).get()
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"elapsed_ms": elapsed_ms}

    print("ContiguousSender: Registers once, reuses same handle")
    return ContiguousReceiver, ContiguousSender


@app.cell
def _(mo):
    mo.md(r"""
    ### Pattern 3: Scattered + RDMAAction

    Register each buffer at init, build a transfer plan once via handshake:

    **What is RDMAAction?**

    Think of `RDMAAction` as a **transfer plan**. You describe all the reads/writes you want
    to do, then `submit()` executes the whole plan at once:

    ```python
    # Build the plan once
    action = RDMAAction()
    action.read_into(handle1, local_buffer1)
    action.read_into(handle2, local_buffer2)
    action.read_into(handle3, local_buffer3)

    # Execute whenever you want - just one call
    action.submit().get()
    ```

    This is useful when you have many scattered buffers (like model parameters) and want
    to batch them into a single logical operation.
    """)
    return


@app.cell
def _(Actor, RDMAAction, RDMABuffer, current_rank, endpoint, time, torch):
    class ScatteredSender(Actor):
        """Multiple buffers, each registered once at startup."""

        def __init__(self, layer_sizes: list):
            self.layer_sizes = layer_sizes
            self.layers = []
            self.handles = []

            for i, size in enumerate(layer_sizes):
                layer = torch.zeros(size, dtype=torch.float32)
                layer.fill_(float(i + 1))
                self.layers.append(layer)
                # Register ONCE at startup
                byte_view = layer.view(torch.uint8).flatten()
                self.handles.append(RDMABuffer(byte_view))

        @endpoint
        def get_handles(self) -> list:
            return [(size, handle) for size, handle in zip(self.layer_sizes, self.handles)]

    class ScatteredReceiver(Actor):
        """Receives from scattered sender with RDMAAction batching."""

        def __init__(self, layer_sizes: list):
            self.layer_sizes = layer_sizes
            self.layers = [torch.zeros(size, dtype=torch.float32) for size in layer_sizes]
            self.rank = current_rank().rank
            self.action = None  # Built on handshake

        @endpoint
        def handshake(self, sender: ScatteredSender):
            """Call once to build the RDMAAction transfer plan."""
            handles = sender.get_handles.call_one().get()
            self.action = RDMAAction()
            for i, (size, handle) in enumerate(handles):
                byte_view = self.layers[i].view(torch.uint8).flatten()
                self.action.read_into(handle, byte_view)
            return "Transfer plan ready"

        @endpoint
        def receive_step(self) -> dict:
            """Execute the cached transfer plan."""
            start = time.perf_counter()
            self.action.submit().get()
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"elapsed_ms": elapsed_ms}

    print("ScatteredSender: Registers each layer once")
    print("ScatteredReceiver: handshake() builds plan, receive_step() executes it")
    return ScatteredReceiver, ScatteredSender


@app.cell
def _(mo):
    mo.md(r"""
    ### Running the Benchmark
    """)
    return


@app.cell
def _(
    ContiguousReceiver,
    ContiguousSender,
    NaiveReceiver,
    NaiveSender,
    ScatteredReceiver,
    ScatteredSender,
    is_rdma_available,
    this_host,
):
    def run_benchmark():
        """Compare the three approaches over multiple steps."""
        if not is_rdma_available():
            print("⚠ RDMA not available on this machine. Skipping benchmark.")
            print("  Run on an RDMA-capable host to see real results.")
            return None

        layer_sizes = [10000, 50000, 20000]  # 80000 floats total
        total_size = sum(layer_sizes)
        num_steps = 5

        host = this_host()
        sender_procs = host.spawn_procs(per_host={"procs": 1})
        receiver_procs = host.spawn_procs(per_host={"procs": 1})

        # Spawn all actors
        naive_sender = sender_procs.spawn("naive_s", NaiveSender, layer_sizes)
        naive_receiver = receiver_procs.spawn("naive_r", NaiveReceiver, layer_sizes)

        cont_sender = sender_procs.spawn("cont_s", ContiguousSender, layer_sizes)
        cont_receiver = receiver_procs.spawn("cont_r", ContiguousReceiver, total_size)

        scat_sender = sender_procs.spawn("scat_s", ScatteredSender, layer_sizes)
        scat_receiver = receiver_procs.spawn("scat_r", ScatteredReceiver, layer_sizes)

        # ── WARMUP ──
        # Force the full RDMA initialization chain (RdmaController spawn,
        # RdmaManagerActor spawn, QP establishment) so benchmarks measure
        # only what we intend to measure.
        print("Warming up RDMA subsystem...")
        naive_receiver.warmup.call_one(naive_sender).get()
        cont_receiver.warmup.call_one(cont_sender).get()
        scat_receiver.handshake.call_one(scat_sender).get()
        scat_receiver.receive_step.call_one().get()  # warm the transfer path
        print("Warmup complete.\n")

        print("=== RDMA Buffer Pattern Benchmark ===")
        print(f"Transferring {total_size} floats ({total_size * 4 / 1024:.1f} KB) x {num_steps} steps\n")

        results = {}

        # Pattern 1: Naive (re-register each step)
        times = []
        for step in range(num_steps):
            r = naive_receiver.receive_step.call_one(naive_sender).get()
            times.append(r["elapsed_ms"])
        results["Naive"] = times
        print(f"Pattern 1 — Naive (re-register each step):")
        for i, t in enumerate(times):
            print(f"  Step {i+1}: {t:.2f}ms")
        print(f"  Average: {sum(times)/len(times):.2f}ms\n")

        # Pattern 2: Contiguous (one buffer, one read)
        times = []
        for step in range(num_steps):
            r = cont_receiver.receive_step.call_one(cont_sender).get()
            times.append(r["elapsed_ms"])
        results["Contiguous"] = times
        print(f"Pattern 2 — Contiguous (one buffer, one read):")
        for i, t in enumerate(times):
            print(f"  Step {i+1}: {t:.2f}ms")
        print(f"  Average: {sum(times)/len(times):.2f}ms\n")

        # Pattern 3: Scattered + RDMAAction (batched plan)
        scat_receiver.handshake.call_one(scat_sender).get()  # rebuild plan
        times = []
        for step in range(num_steps):
            r = scat_receiver.receive_step.call_one().get()
            times.append(r["elapsed_ms"])
        results["Scattered"] = times
        print(f"Pattern 3 — Scattered + RDMAAction (batched plan):")
        for i, t in enumerate(times):
            print(f"  Step {i+1}: {t:.2f}ms")
        print(f"  Average: {sum(times)/len(times):.2f}ms\n")

        print("=== What's Happening ===")
        print("Naive: Re-registers MRs + sequential per-layer reads (~9ms)")
        print("Contiguous: One MR, one read — fewest round trips (~3ms)")
        print("Scattered + RDMAAction: Pre-built transfer plan (~8ms, Python overhead)")

        return results

    benchmark_results = run_benchmark()
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What the benchmark shows:**

    - **Naive** (~9ms): Re-registers MRs each step, plus sequential per-layer reads.
      At larger buffer sizes the MR cost grows, but here it's masked by Python overhead.
    - **Contiguous** (~3ms): One MR, one `read_into()` — fewest round trips wins.
      The trade-off: in practice, model parameters aren't contiguous, so you'd need
      to copy into a contiguous staging buffer first.
    - **Scattered + RDMAAction** (~8ms): Pre-built transfer plan, works with non-contiguous
      layouts. Currently has Python overhead from the batching logic; lowering `RDMAAction`
      into Rust will close the gap with Contiguous.

    The key lesson: **minimize round trips.** Whether that means packing into one buffer
    or batching via `RDMAAction`, the goal is the same — fewer Python-to-Rust-to-NIC
    transitions per sync.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    We started with a question — *where do the milliseconds go?* — and traced the answer
    through two layers:

    1. **ibverbs internals** — QPs, MRs, CQs: the primitives that make RDMA work, and how
       Monarch's actor model wraps them into composable connections
    2. **RDMA buffer patterns** — From naive re-registration to contiguous buffers
       and batched transfer plans via `RDMAAction` — minimizing Python round trips

    We focused on **ibverbs** specifically because that's what Monarch's RDMA subsystem
    supports today (InfiniBand and RoCE via Mellanox/NVIDIA ConnectX NICs). EFA (Elastic
    Fabric Adapter, used on AWS) is a relevant transport but not yet supported — it's
    actively being worked on.

    The main notebook (07) covers the high-level patterns for async RL weight sync —
    magic pointers, CPU staging, circular buffers, and DTensor re-sharding. This notebook
    showed the RDMA specifics underneath: what the NIC is actually doing and how to avoid
    paying unnecessary costs on the hot path.

    ---

    **Previous:** [NB07 — RDMA Weight Sync](./07_rdma_weight_sync.html) · **Next:** [NB08 — Async RL E2E](./08_rl_e2e.html)
    """)
    return


if __name__ == "__main__":
    app.run()
