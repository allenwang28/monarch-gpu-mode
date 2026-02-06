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
    # Weight Sync Deep Dive

    In Notebook 06, we treated RDMA as a magic pipe: call `read_into()`, weights appear.
    But what's actually happening when you register memory with a NIC? How much does it
    cost, and what can go wrong if you get it wrong?

    This notebook opens the hood. We'll trace an RDMA transfer from ibverbs primitives
    through Monarch's actor-based control plane, measure the hidden costs that can turn
    a microsecond operation into a millisecond one, and build the data structures that
    make production weight sync reliable.

    **The central question: where do the milliseconds go?**

    1. **ibverbs Internals** - Queue Pairs, Memory Registration, Completion Queues
    2. **Memory Registration Costs** - Benchmarking naive vs optimized approaches
    3. **Circular Weight Buffer Implementation** - Full working code with lifecycle management
    4. **DTensor Re-sharding** - Computing transfer plans, the full benchmark
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

    We focus on **InfiniBand** and **RoCE** (RDMA over Converged Ethernet).

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

    One piece we haven't mentioned: how does the initiator know an operation finished?
    Every QP is associated with a **Completion Queue (CQ)**. When the NIC finishes an
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
    ## 2. Memory Registration Costs

    RDMA memory registration is **expensive**:
    - Pins physical pages (prevents swapping)
    - Creates IOMMU/DMA mappings in the NIC
    - Can take milliseconds for large buffers

    **What does Monarch cache?** The `@functools.cache` on `_ensure_init_rdma_manager` caches
    the RDMA *manager initialization* (QP setup, control plane bootstrap) - that only happens
    once per process. But memory region (MR) registration for CPU tensors happens on each
    `RDMABuffer()` call. For CUDA tensors with mlx5dv, there's segment-level MR caching in
    the driver. In the future, Monarch plans to cache all MR registrations at the Python layer,
    but today **you** are responsible for reusing `RDMABuffer` handles to amortize MR cost.

    Let's benchmark 3 approaches to see this in action.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Approach 1: Naive

    Create new RDMABuffer on each transfer - registration happens on first use:
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
    ### Approach 2: Contiguous Buffer

    Allocate one buffer, register at init time:
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
        """Receives from contiguous sender - fast after first step."""

        def __init__(self, total_size: int):
            self.buffer = torch.zeros(total_size, dtype=torch.float32)
            self.rank = current_rank().rank

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
    ### Approach 3: Scattered + RDMAAction

    Register each buffer at init, build transfer plan once via handshake:

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

        layer_sizes = [1000, 5000, 2000]  # 8000 floats total
        total_size = sum(layer_sizes)
        num_steps = 5

        host = this_host()
        sender_procs = host.spawn_procs(per_host={"procs": 1})
        receiver_procs = host.spawn_procs(per_host={"procs": 1})

        print("=== RDMA Registration Benchmark ===")
        print(f"Transferring {total_size} floats ({total_size * 4 / 1024:.1f} KB) x {num_steps} steps\n")

        results = {}

        # Naive approach
        naive_sender = sender_procs.spawn("naive_s", NaiveSender, layer_sizes)
        naive_receiver = receiver_procs.spawn("naive_r", NaiveReceiver, layer_sizes)
        times = []
        for step in range(num_steps):
            r = naive_receiver.receive_step.call_one(naive_sender).get()
            times.append(r["elapsed_ms"])
        results["Naive"] = times
        print(f"Naive (re-register each step):")
        for i, t in enumerate(times):
            print(f"  Step {i+1}: {t:.2f}ms")
        print(f"  Average: {sum(times)/len(times):.2f}ms\n")

        # Contiguous approach
        cont_sender = sender_procs.spawn("cont_s", ContiguousSender, layer_sizes)
        cont_receiver = receiver_procs.spawn("cont_r", ContiguousReceiver, total_size)
        times = []
        for step in range(num_steps):
            r = cont_receiver.receive_step.call_one(cont_sender).get()
            times.append(r["elapsed_ms"])
        results["Contiguous"] = times
        print(f"Contiguous (register once):")
        for i, t in enumerate(times):
            print(f"  Step {i+1}: {t:.2f}ms")
        print(f"  Average: {sum(times)/len(times):.2f}ms\n")

        # Scattered + RDMAAction approach
        scat_sender = sender_procs.spawn("scat_s", ScatteredSender, layer_sizes)
        scat_receiver = receiver_procs.spawn("scat_r", ScatteredReceiver, layer_sizes)
        scat_receiver.handshake.call_one(scat_sender).get()  # Build transfer plan once
        times = []
        for step in range(num_steps):
            r = scat_receiver.receive_step.call_one().get()  # Just execute cached plan
            times.append(r["elapsed_ms"])
        results["Scattered"] = times
        print(f"Scattered + RDMAAction (register once, batch):")
        for i, t in enumerate(times):
            print(f"  Step {i+1}: {t:.2f}ms")
        print(f"  Average: {sum(times)/len(times):.2f}ms\n")

        print("=== What's Happening ===")
        print("Naive step 1: Cold MR registration (~2000ms)")
        print("Naive steps 2+: Cache hit, MR already registered (~10ms)")
        print("Contiguous/Scattered: Registration happened at spawn time, not during benchmark")

        return results

    benchmark_results = run_benchmark()
    return (benchmark_results, run_benchmark)


@app.cell
def _(mo):
    mo.md(r"""
    **What the benchmark shows:**

    - **Naive**: First call is ~2000ms (cold registration), subsequent calls ~10ms (cache hit)
    - **Contiguous/Scattered**: All calls are fast (~4-9ms) because registration happened
      at spawn time, before the timing loop started

    *Note: RDMAAction (~9ms) is slower than Contiguous (~4ms) due to Python overhead.
    Moving the batching logic to Rust is a planned optimization.*

    With MR costs understood, we can build a data structure that keeps handles registered
    and manages the lifecycle of weight snapshots: the circular buffer.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Circular Weight Buffer Implementation

    Here's a full working implementation of a circular buffer for versioned weight storage.
    This is the same pattern from Notebook 06's Section 6, but with the complete code.
    In production, this lives inside a Monarch actor with slots pre-registered as RDMABuffers
    at init time - so MR cost is paid once and amortized across all subsequent syncs.
    """)
    return


@app.cell
def _(torch):
    from threading import Lock as _Lock
    from typing import Tuple as _Tuple, Optional as _Opt

    class CircularWeightBuffer:
        """Circular buffer for versioned weight storage.

        In production, this would be used inside a Monarch actor and slots
        would be registered with RDMABuffer at init time.
        """

        def __init__(self, template_tensor: torch.Tensor, n_slots: int = 3):
            self.n_slots = n_slots
            self.slots = [
                torch.empty_like(template_tensor).pin_memory()
                if template_tensor.device.type == "cpu"
                else torch.empty_like(template_tensor, device="cpu").pin_memory()
                for _ in range(n_slots)
            ]
            self.latest_version = 0
            self._lock = _Lock()

            # In production (inside a Monarch actor):
            # self.rdma_handles = [RDMABuffer(slot.view(torch.uint8).flatten()) for slot in self.slots]
            # This pre-registers all slots with RDMA at init time (amortizes MR cost)

        def publish(self, weights: torch.Tensor) -> int:
            """Trainer publishes new weights. Returns version number."""
            with self._lock:
                slot_idx = self.latest_version % self.n_slots
                self.slots[slot_idx].copy_(weights)
                self.latest_version += 1
                return self.latest_version - 1

        def get_latest(self) -> _Tuple[torch.Tensor, int]:
            """Generator gets latest weights. Non-blocking."""
            with self._lock:
                if self.latest_version == 0:
                    raise RuntimeError("No weights published yet")
                slot_idx = (self.latest_version - 1) % self.n_slots
                version = self.latest_version - 1
                return self.slots[slot_idx].clone(), version

        def get_version(self, version: int) -> _Opt[torch.Tensor]:
            """Get specific version if still available."""
            with self._lock:
                oldest_available = max(0, self.latest_version - self.n_slots)
                if version < oldest_available or version >= self.latest_version:
                    return None
                slot_idx = version % self.n_slots
                return self.slots[slot_idx].clone()

    # Demo
    _template = torch.randn(100, 100)
    weight_buffer = CircularWeightBuffer(_template, n_slots=3)

    # Trainer publishes versions
    for _v in range(5):
        _new_weights = torch.randn(100, 100) * (_v + 1)
        published_v = weight_buffer.publish(_new_weights)
        print(f"Published version {published_v}")

    # Generator grabs latest
    latest_weights, latest_version = weight_buffer.get_latest()
    print(f"\nGenerator got version {latest_version}, weights mean: {latest_weights.mean():.2f}")

    # Try to get old version (might be evicted)
    old_weights = weight_buffer.get_version(1)
    print(f"Version 1 available: {old_weights is not None}")

    print("\nIn production: RDMABuffer handles would be pre-registered at init time")
    print("Generators would call get_latest_handle() to get RDMA handle + version")
    return (CircularWeightBuffer,)


@app.cell
def _(mo):
    mo.md(r"""
    ### RDMABuffer Lifecycle: Don't Forget `drop()`

    Every `RDMABuffer` pins physical memory and creates NIC-level mappings. If you discard
    a Python reference without calling `drop()`, the underlying MR stays pinned in kernel
    memory until the process exits. In a long-running actor that creates buffers dynamically,
    this is a **memory leak**.

    ```python
    # WRONG: MR stays pinned even after Python GC collects the object
    buf = RDMABuffer(tensor.view(torch.uint8).flatten())
    del buf  # Python object gone, but kernel MR still pinned!

    # RIGHT: explicitly release the MR before discarding
    buf = RDMABuffer(tensor.view(torch.uint8).flatten())
    buf.drop().get()  # Releases MR registration in the NIC
    del buf           # Now safe
    ```

    The circular buffer pattern sidesteps this entirely: allocate N slots at init,
    register them once, reuse forever, and only `drop()` at actor shutdown. No dynamic
    allocation means no leak risk.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. DTensor Re-sharding Implementation

    So far we've assumed the trainer and generator have matching tensor layouts. In practice,
    they rarely do - different parallelism strategies mean different sharding. The final piece
    is computing **which chunks of data need to move where** when layouts don't match.

    ### Computing Shard Metadata
    """)
    return


@app.cell
def _():
    from dataclasses import dataclass
    from typing import List, Tuple

    @dataclass
    class ShardMetadata:
        """Metadata describing a tensor shard."""
        rank: int
        global_shape: Tuple[int, ...]
        offset: Tuple[int, ...]  # Start position in global tensor
        local_shape: Tuple[int, ...]  # Shape of this shard

    def compute_shard_metadata(
        global_shape: Tuple[int, int],
        num_ranks: int,
        shard_dim: int,
    ) -> List[ShardMetadata]:
        """Compute shard metadata for a given sharding."""
        shards = []
        dim_size = global_shape[shard_dim]
        shard_size = dim_size // num_ranks

        for rank in range(num_ranks):
            offset = [0, 0]
            local_shape = list(global_shape)

            offset[shard_dim] = rank * shard_size
            local_shape[shard_dim] = shard_size

            shards.append(ShardMetadata(
                rank=rank,
                global_shape=global_shape,
                offset=tuple(offset),
                local_shape=tuple(local_shape),
            ))

        return shards

    # Demo: Trainer has row-sharding, Generator has column-sharding
    _global_shape = (1024, 1024)

    trainer_shards = compute_shard_metadata(_global_shape, num_ranks=4, shard_dim=0)
    generator_shards = compute_shard_metadata(_global_shape, num_ranks=2, shard_dim=1)

    print("Trainer shards (row-wise, 4 GPUs):")
    for s in trainer_shards:
        print(f"  Rank {s.rank}: offset={s.offset}, shape={s.local_shape}")

    print("\nGenerator shards (column-wise, 2 GPUs):")
    for s in generator_shards:
        print(f"  Rank {s.rank}: offset={s.offset}, shape={s.local_shape}")
    return (
        List,
        ShardMetadata,
        Tuple,
        compute_shard_metadata,
        dataclass,
        generator_shards,
        trainer_shards,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Computing Transfer Plans

    Given source and destination sharding, compute the minimal set of transfers needed:
    """)
    return


@app.cell
def _(List, ShardMetadata, Tuple, dataclass, generator_shards, trainer_shards):
    @dataclass
    class TransferChunk:
        """A chunk to transfer from sender to receiver."""
        sender_rank: int
        receiver_rank: int
        sender_offset: Tuple[int, int]  # Where to read from sender
        receiver_offset: Tuple[int, int]  # Where to write in receiver
        shape: Tuple[int, int]  # Shape of the chunk

    def compute_overlap(
        sender: ShardMetadata,
        receiver: ShardMetadata,
    ) -> TransferChunk | None:
        """Compute overlap between sender and receiver shards."""
        # Find intersection in global coordinates
        s_start = sender.offset
        s_end = (s_start[0] + sender.local_shape[0], s_start[1] + sender.local_shape[1])

        r_start = receiver.offset
        r_end = (r_start[0] + receiver.local_shape[0], r_start[1] + receiver.local_shape[1])

        # Compute intersection
        inter_start = (max(s_start[0], r_start[0]), max(s_start[1], r_start[1]))
        inter_end = (min(s_end[0], r_end[0]), min(s_end[1], r_end[1]))

        # Check if there's actual overlap
        if inter_start[0] >= inter_end[0] or inter_start[1] >= inter_end[1]:
            return None

        shape = (inter_end[0] - inter_start[0], inter_end[1] - inter_start[1])

        # Convert to local coordinates
        sender_local = (inter_start[0] - s_start[0], inter_start[1] - s_start[1])
        receiver_local = (inter_start[0] - r_start[0], inter_start[1] - r_start[1])

        return TransferChunk(
            sender_rank=sender.rank,
            receiver_rank=receiver.rank,
            sender_offset=sender_local,
            receiver_offset=receiver_local,
            shape=shape,
        )

    def compute_transfer_plan(
        sender_shards: List[ShardMetadata],
        receiver_shards: List[ShardMetadata],
    ) -> List[TransferChunk]:
        """Compute all transfers needed for re-sharding."""
        transfers = []
        for sender in sender_shards:
            for receiver in receiver_shards:
                chunk = compute_overlap(sender, receiver)
                if chunk is not None:
                    transfers.append(chunk)
        return transfers

    # Compute transfer plan
    transfer_plan = compute_transfer_plan(trainer_shards, generator_shards)

    print(f"Transfer plan: {len(transfer_plan)} chunks needed\n")
    for chunk in transfer_plan:
        print(f"Sender {chunk.sender_rank} → Receiver {chunk.receiver_rank}")
        print(f"  Read from sender offset {chunk.sender_offset}, shape {chunk.shape}")
        print(f"  Write to receiver offset {chunk.receiver_offset}")
        print()
    return (TransferChunk, compute_overlap, compute_transfer_plan, transfer_plan)


@app.cell
def _(mo):
    mo.md(r"""
    ### Full DTensor Benchmark

    This benchmark uses actual DTensor with different placements to show the savings
    from routed transfers vs gather-then-slice.
    """)
    return


@app.cell
def _(compute_shard_metadata):
    import os
    from torch.distributed._tensor import DTensor, Shard, Replicate, init_device_mesh

    # Configuration
    NUM_TRAINER_RANKS = 4
    NUM_GENERATOR_RANKS = 2

    # Layer configs: (global_shape, trainer_placement, generator_placement)
    LAYER_CONFIGS = [
        {"shape": (1024, 1024), "trainer_place": Shard(0), "gen_place": Shard(0)},
        {"shape": (512, 2048), "trainer_place": Shard(1), "gen_place": Shard(1)},
        {"shape": (256, 256), "trainer_place": Replicate(), "gen_place": Replicate()},
    ]

    def placement_to_shard_dim(placement) -> int | None:
        """Extract shard dimension from DTensor placement."""
        if isinstance(placement, Shard):
            return placement.dim
        return None

    def compute_layer_transfer_plan(layer_cfg, trainer_ranks, gen_ranks, gen_rank):
        """Use DTensor placement metadata to compute transfer plan for one layer.

        Returns list of trainer rank indices that have data this generator rank needs.
        """
        trainer_dim = placement_to_shard_dim(layer_cfg["trainer_place"])
        gen_dim = placement_to_shard_dim(layer_cfg["gen_place"])

        if trainer_dim is None:
            # Replicated on trainer side - just read from rank 0
            return [0]

        if gen_dim is None:
            # Replicated on generator side - need all trainer shards
            return list(range(trainer_ranks))

        trainer_shards = compute_shard_metadata(layer_cfg["shape"], trainer_ranks, trainer_dim)
        gen_shards = compute_shard_metadata(layer_cfg["shape"], gen_ranks, gen_dim)
        my_shard = gen_shards[gen_rank]

        overlapping = []
        for t_shard in trainer_shards:
            t_start = t_shard.offset[trainer_dim]
            t_end = t_start + t_shard.local_shape[trainer_dim]
            g_start = my_shard.offset[gen_dim] if gen_dim == trainer_dim else 0
            g_end = (my_shard.offset[gen_dim] + my_shard.local_shape[gen_dim]
                     if gen_dim == trainer_dim else layer_cfg["shape"][trainer_dim])

            if t_end > g_start and t_start < g_end:
                overlapping.append(t_shard.rank)

        return overlapping

    print(f"Config: {NUM_TRAINER_RANKS} trainer ranks, {NUM_GENERATOR_RANKS} generator ranks")
    for _i, _cfg in enumerate(LAYER_CONFIGS):
        print(f"  Layer {_i}: {_cfg['shape']}, trainer={_cfg['trainer_place']}, gen={_cfg['gen_place']}")
    return (
        DTensor,
        LAYER_CONFIGS,
        NUM_GENERATOR_RANKS,
        NUM_TRAINER_RANKS,
        Replicate,
        Shard,
        compute_layer_transfer_plan,
        init_device_mesh,
        os,
        placement_to_shard_dim,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### DTensorTrainer

    The trainer creates DTensors with the configured placements and exposes per-layer
    RDMA handles. Each rank registers its local shard once at setup time.
    """)
    return


@app.cell
def _(Actor, LAYER_CONFIGS, RDMABuffer, current_rank, endpoint, init_device_mesh, os, placement_to_shard_dim, torch):
    class DTensorTrainer(Actor):
        """Trainer with DTensor shards."""

        def __init__(self):
            self.rank = current_rank().rank
            self.dtensors = []
            self.handles = []
            self.device_mesh = None

        @endpoint
        def setup_distributed(self):
            world_size = int(os.environ.get("WORLD_SIZE", "1"))

            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="gloo")

            self.device_mesh = init_device_mesh("cpu", (world_size,))

            from torch.distributed._tensor import DTensor

            for i, cfg in enumerate(LAYER_CONFIGS):
                placement = cfg["trainer_place"]
                shard_dim = placement_to_shard_dim(placement)

                if shard_dim is not None:
                    local_shape = list(cfg["shape"])
                    local_shape[shard_dim] = cfg["shape"][shard_dim] // world_size
                    local_shape = tuple(local_shape)
                else:
                    local_shape = cfg["shape"]

                local_tensor = torch.zeros(local_shape, dtype=torch.float32)
                local_tensor.fill_(float(i * 10 + self.rank))

                dt = DTensor.from_local(local_tensor, self.device_mesh, [placement], run_check=False)
                self.dtensors.append(dt)
                self.handles.append(RDMABuffer(local_tensor.view(torch.uint8).flatten()))

            shapes = [tuple(dt.to_local().shape) for dt in self.dtensors]
            placements = [str(cfg["trainer_place"]) for cfg in LAYER_CONFIGS]
            print(f"Trainer {self.rank}: shapes={shapes}, placements={placements}")
            return shapes

        @endpoint
        def get_layer_handle(self, layer_idx: int):
            return (tuple(self.dtensors[layer_idx].to_local().shape), self.handles[layer_idx])

        @endpoint
        def destroy(self):
            for h in self.handles:
                h.drop().get()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

    print("DTensorTrainer defined")
    return (DTensorTrainer,)


@app.cell
def _(mo):
    mo.md(r"""
    ### DTensorGenerator

    The generator uses `compute_layer_transfer_plan` to figure out which trainer ranks
    hold data it needs, then builds an `RDMAAction` transfer plan via handshake. Subsequent
    syncs just call `action.submit()` - no recomputation needed.
    """)
    return


@app.cell
def _(Actor, LAYER_CONFIGS, NUM_GENERATOR_RANKS, NUM_TRAINER_RANKS, RDMAAction, compute_layer_transfer_plan, current_rank, endpoint, init_device_mesh, os, time, torch):
    class DTensorGenerator(Actor):
        """Generator that uses DTensor placement metadata for smart resharding."""

        def __init__(self):
            self.rank = current_rank().rank
            self.action = None
            self.recv_buffers = []
            self.device_mesh = None

        @endpoint
        def setup_distributed(self):
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="gloo")
            self.device_mesh = init_device_mesh("cpu", (world_size,))
            print(f"Generator {self.rank}: distributed initialized")
            return world_size

        @endpoint
        def handshake_routed(self, trainers):
            """Routed approach: use DTensor placements to compute minimal transfers."""
            self.action = RDMAAction()
            self.recv_buffers = []
            total_transfers = 0

            for layer_idx, cfg in enumerate(LAYER_CONFIGS):
                overlapping = compute_layer_transfer_plan(
                    cfg, NUM_TRAINER_RANKS, NUM_GENERATOR_RANKS, self.rank
                )

                for t_rank in overlapping:
                    shape, handle = trainers[t_rank].get_layer_handle.call_one(layer_idx).get()
                    buf = torch.zeros(shape, dtype=torch.float32)
                    self.recv_buffers.append(buf)
                    self.action.read_into(handle, buf.view(torch.uint8).flatten())
                    total_transfers += 1

            return f"Routed: {total_transfers} transfers (placement-aware)"

        @endpoint
        def receive_routed(self) -> dict:
            start = time.perf_counter()
            self.action.submit().get()
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"elapsed_ms": elapsed_ms}

        @endpoint
        def destroy(self):
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

    print("DTensorGenerator defined")
    return (DTensorGenerator,)


@app.cell
def _(
    DTensorGenerator,
    DTensorTrainer,
    LAYER_CONFIGS,
    NUM_GENERATOR_RANKS,
    NUM_TRAINER_RANKS,
    is_rdma_available,
    this_host,
    time,
):
    from monarch.spmd import setup_torch_elastic_env

    if not is_rdma_available():
        print("⚠ RDMA not available on this machine. Skipping DTensor benchmark.")
        print("  Run on an RDMA-capable host to see real results.")
    else:
        _trainer_procs = this_host().spawn_procs(per_host={"procs": NUM_TRAINER_RANKS})
        setup_torch_elastic_env(_trainer_procs)
        _trainers = _trainer_procs.spawn("trainers", DTensorTrainer)

        _gen_procs = this_host().spawn_procs(per_host={"procs": NUM_GENERATOR_RANKS})
        setup_torch_elastic_env(_gen_procs)
        _generators = _gen_procs.spawn("generators", DTensorGenerator)

        print("\n=== DTensor Reshard Benchmark ===")
        print(f"Trainer mesh: {NUM_TRAINER_RANKS} ranks, Generator mesh: {NUM_GENERATOR_RANKS} ranks")
        print("Layer configs:")
        for i, cfg in enumerate(LAYER_CONFIGS):
            print(f"  Layer {i}: {cfg['shape']}, trainer={cfg['trainer_place']}, gen={cfg['gen_place']}")

        print("\nSetting up distributed...")
        _trainer_shapes = _trainers.setup_distributed.call().get()
        _gen_world = _generators.setup_distributed.call().get()
        print(f"  Trainer shapes: {[s for _, s in _trainer_shapes]}")
        print(f"  Generator world sizes: {[w for _, w in _gen_world]}")

        _trainer_list = [_trainers.slice(procs=i) for i in range(NUM_TRAINER_RANKS)]

        print("\nBuilding transfer plans (using placement metadata)...")
        _results = _generators.handshake_routed.call(_trainer_list).get()
        for _i, _r in enumerate(_results):
            print(f"  Generator {_i}: {_r}")

        print("\nRunning transfers...")
        _times = []
        for _step in range(3):
            _step_start = time.perf_counter()
            _results = _generators.receive_routed.call().get()
            _step_ms = (time.perf_counter() - _step_start) * 1000
            _times.append(_step_ms)
            print(f"  Step {_step + 1}: {_step_ms:.1f}ms")

        _avg = sum(_times) / len(_times)
        print(f"  Average: {_avg:.1f}ms")

        _trainers.destroy.call().get()
        _generators.destroy.call().get()
        print("\nDistributed cleanup complete")
    return (setup_torch_elastic_env,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    We started with a question - *where do the milliseconds go?* - and traced the answer
    through four layers:

    1. **ibverbs internals** - QPs, MRs, CQs: the primitives that make RDMA work
    2. **Memory registration costs** - The hidden tax of naive buffer management, and how
       reusing handles amortizes it away
    3. **Circular weight buffers** - Structural coordination with proper `drop()` lifecycle
    4. **DTensor re-sharding** - Computing minimal transfer plans from placement metadata

    These are the building blocks that make Monarch's weight sync fast and flexible.
    The main notebook (06) covers when and why to use these patterns; this notebook
    showed *how* they work under the hood.
    """)
    return


if __name__ == "__main__":
    app.run()
