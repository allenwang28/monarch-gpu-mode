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
    import torch
    from monarch.actor import Actor, endpoint, current_rank, this_host
    return Actor, current_rank, endpoint, this_host, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Interactive DevX: Monarch as Remote Torchrun
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Pain of Traditional Development

    ```
    Traditional Distributed Development Loop:

    Developer                    SLURM                     Cluster
        │                          │                          │
        ├── sbatch job.sh ────────►│                          │
        │                          ├── queue... ──────────────►
        │                          │   (minutes to hours)     │
        │                          │◄─────────────────────────┤
        │◄── job started ──────────┤                          │
        │                          │                          │
        │   ...wait for completion...                         │
        │                          │                          │
        ├── cat slurm-*.out ──────►│                          │
        │◄── scattered logs ───────┤                          │
        │                          │                          │
        │   "Found bug on line 42"                            │
        │                          │                          │
        └── sbatch again... ──────►│   (repeat forever)       │
    ```

    **Key problems:**

    - Queue wait time dominates iteration time
    - Logs scattered across nodes
    - Each fix requires full resubmission
    - No interactive debugging
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Monarch Solution

    ```
    Monarch Development Loop:

    Developer                    Monarch                   Cluster
        │                          │                          │
        ├── allocate hosts ───────►│                          │
        │   (one time, slow)       ├── gang schedule ────────►│
        │                          │◄─────────────────────────┤
        │◄── HostMesh ready ───────┤                          │
        │                          │                          │
        │   === Fast iteration loop ===                       │
        │                          │                          │
        ├── spawn_procs() ────────►│   (instant)              │
        ├── spawn actors ─────────►│   (instant)              │
        ├── call endpoints ───────►│   (instant)              │
        │◄── aggregated logs ──────┤                          │
        │                          │                          │
        │   "Found bug, fixing..."                            │
        │                          │                          │
        ├── spawn_procs() again ──►│   (instant, same hosts!) │
        └── ...                    │                          │
    ```

    **Key insight:** Allocation is slow, but spawning on existing hosts is fast.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What you'll learn:**

    1. The pain of traditional distributed development (sbatch, wait, debug, repeat)
    2. `this_host()` for local development, Monarch's `Job` for real clusters
    3. Mesh operations: `.call()`, `.slice()`, `.call_one()`
    4. How logs and errors are aggregated back to your controller
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Your First Interactive Session

    For local development, `this_host()` gives you a host that can spawn multiple
    processes. On a real cluster, you'd use something like `SlurmJob` to allocate hosts — but the
    pattern is identical.

    *Note: `"gpus"` in `per_host={"gpus": N}` is a dimension label for the mesh —
    it doesn't require physical GPUs. You can run these examples on a CPU-only
    machine.*
    """)
    return


@app.cell
def _(Actor, current_rank, endpoint, this_host, torch):
    class Worker(Actor):
        """A simple worker that knows its rank."""

        def __init__(self):
            self.rank = current_rank().rank
            print(f"Worker initialized on rank {self.rank}")

        @endpoint
        def compute(self, data: torch.Tensor) -> dict:
            result = data.sum().item() * (self.rank + 1)
            return {
                "rank": self.rank,
                "input_shape": tuple(data.shape),
                "result": result,
            }

    # Spawn 4 worker processes locally
    host = this_host()
    procs = host.spawn_procs(per_host={"gpus": 4})
    workers = procs.spawn("workers", Worker)

    # Call all workers with the same data
    data = torch.randn(10, 10)
    results = workers.compute.call(data).get()

    for _point, _result in results.items():
        print(f"Worker {_result['rank']}: computed {_result['result']:.2f}")
    return (workers,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Working with Individual Actors

    When you spawn actors on a mesh, you get an ActorMesh.
    You can slice it to talk to individual actors or subsets.
    """)
    return


@app.cell
def _(torch, workers):
    # Call all workers
    all_results = workers.compute.call(torch.ones(5, 5)).get()
    print(f"All workers returned: {list(all_results.values())}")

    # Call just worker 0
    worker_0 = workers.slice(gpus=0)
    result_0 = worker_0.compute.call_one(torch.ones(5, 5)).get()
    print(f"Worker 0 alone: {result_0}")

    # Call workers 1 and 2
    subset = workers.slice(gpus=slice(1, 3))
    subset_results = subset.compute.call(torch.ones(5, 5)).get()
    print(f"Workers 1-2 returned: {list(subset_results.values())}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## All Logs in One Place

    When actors print, their output is aggregated back to your controller — no
    SSH-ing into nodes to check logs.
    """)
    return


@app.cell
def _(Actor, current_rank, endpoint, this_host):
    import time

    class VerboseWorker(Actor):
        def __init__(self):
            self.rank = current_rank().rank
            print(f"[Rank {self.rank}] Initializing...")

        @endpoint
        def do_work(self, task_id: int) -> str:
            print(f"[Rank {self.rank}] Starting task {task_id}")
            time.sleep(0.1 * (self.rank + 1))  # Simulate varying work
            print(f"[Rank {self.rank}] Completed task {task_id}")
            return f"rank_{self.rank}_task_{task_id}"

    verbose_procs = this_host().spawn_procs(per_host={"gpus": 3})
    verbose_workers = verbose_procs.spawn("verbose", VerboseWorker)

    # All workers work on the same task — watch the interleaved logs
    verbose_results = verbose_workers.do_work.call(42).get()
    for _point, _result in verbose_results.items():
        print(f"  {_result}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## When Things Go Wrong

    Errors from remote actors are brought back to your controller with full tracebacks.
    """)
    return


@app.cell
def _(Actor, current_rank, endpoint, this_host):
    class FlakyWorker(Actor):
        def __init__(self):
            self.rank = current_rank().rank

        @endpoint
        def risky_operation(self, fail_on_rank: int) -> str:
            if self.rank == fail_on_rank:
                raise ValueError(f"Intentional failure on rank {self.rank}!")
            return f"success on rank {self.rank}"

    flaky_procs = this_host().spawn_procs(per_host={"gpus": 3})
    flaky_workers = flaky_procs.spawn("flaky", FlakyWorker)

    # This will fail on rank 1
    try:
        flaky_results = flaky_workers.risky_operation.call(1).get()
    except Exception as e:
        print(f"Caught error: {type(e).__name__}")
        print(f"Message: {e}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## On a Real Cluster

    Everything we just did locally works the same on a SLURM cluster. The only
    difference is how you get your hosts:

    ```python
    from monarch.job import SlurmJob

    # Allocate hosts (slow, one time)
    job = SlurmJob(meshes={"workers": 4}, gpus_per_node=8)
    host_mesh = job.state().workers

    # From here, same pattern as this_host()
    proc_mesh = host_mesh.spawn_procs(per_host={"gpus": 8})
    trainers = proc_mesh.spawn("trainer", TrainerActor)
    trainers.train.call(dataset).get()

    # Iterate without re-allocating
    proc_mesh_2 = host_mesh.spawn_procs(per_host={"gpus": 8})  # Fast!
    ```

    The key insight: `SlurmJob` handles the slow allocation step. Everything after
    that — `spawn_procs`, `spawn`, `call` — is the same API you just used with
    `this_host()`.

    TODO - pointer to `monarchrun` to wrap existing SPMD workloads
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    **Key takeaways:**

    1. **Decouple allocation from iteration**: Allocate hosts once, spawn fast
    2. **`this_host()`** for local dev, **`SlurmJob`** for clusters — same code
    3. **Mesh operations**: `.call()` broadcasts, `.slice()` targets, `.call_one()`
       returns a single value
    4. **Aggregated everything**: Logs and errors come back to your controller

    We just caught a single failure with try/except. But remember those 419
    interruptions from Llama 3 training? When failures hit every 3 hours across
    16,000 GPUs, you need something more systematic. That's next.
    """)
    return


if __name__ == "__main__":
    app.run()
