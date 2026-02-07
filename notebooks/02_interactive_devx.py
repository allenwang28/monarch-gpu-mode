import marimo

__generated_with = "0.19.9"
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

    return (this_host,)


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

    **Key insight:** Allocation is slow, but re-creating the HostMesh from existing hosts is fast.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## DDP with Monarch

    Let's make this concrete. Below is a standard PyTorch DDP training script — no
    Monarch code at all. We'll use Monarch's `SPMDActor` to launch it, just like
    `torchrun` would, but from this notebook.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The Training Script

    This is `train.py` — a vanilla PyTorch DDP script:

    ```python
    import os
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP

    def main():
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        model = nn.Linear(10, 1).cuda()
        ddp_model = DDP(model)

        optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

        for step in range(5):
            inputs = torch.randn(4, 10).cuda()
            outputs = ddp_model(inputs)
            loss = outputs.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"[Rank {rank}] Step {step} loss={loss.item()}")

        dist.destroy_process_group()

    if __name__ == "__main__":
        main()
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Launching DDP via SPMDActor

    `SPMDActor` sets up the torch elastic environment variables (`RANK`, `LOCAL_RANK`,
    `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`) and executes the training script.
    """)
    return


@app.cell
def _(this_host):
    import os as _os
    from monarch.spmd import SPMDActor

    _GPUS_PER_HOST = 4
    _TRAIN_SCRIPT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "train.py")

    # Spawn processes on the local host
    local_proc_mesh = this_host().spawn_procs(per_host={"gpus": _GPUS_PER_HOST})

    # Spawn SPMDActor — it configures torch elastic env vars automatically
    spmd_actors = local_proc_mesh.spawn("_SPMDActor", SPMDActor)

    # Get master address/port from the first actor
    first_values = dict.fromkeys(local_proc_mesh._labels, 0)
    master_addr, master_port = (
        spmd_actors.slice(**first_values).get_host_port.call_one(None).get()
    )

    # Execute training script across all processes
    spmd_actors.main.call(master_addr, master_port, [_TRAIN_SCRIPT]).get()

    print("DDP training completed!")
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Making SPMD Interactive with `serve()` + `run_spmd()`

    The `SPMDActor` approach above works, but Monarch provides a higher-level API
    that makes the interactive loop explicit: `serve()` allocates hosts and launches
    workers once, then `run_spmd()` can be called repeatedly on the same hosts.
    """)
    return


@app.cell
def _():
    import os as _os
    from monarch.job.spmd import serve

    _GPUS_PER_HOST = 4
    _TRAIN_SCRIPT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "train.py")

    # serve() allocates hosts and launches workers (slow, one time)
    job = serve(
        [
            "torchrun",
            f"--nproc-per-node={_GPUS_PER_HOST}",
            "--standalone",
            _TRAIN_SCRIPT,
        ],
        scheduler="local_cwd",
    )

    # run_spmd() executes the training script on the reserved hosts
    job.run_spmd()
    return (job,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Iterate Without Reprovisioning

    Edit `train.py`, then call `run_spmd()` again — same hosts, no reallocation:
    """)
    return


@app.cell
def _(job):
    # After editing train.py, just re-run on the same hosts
    job.run_spmd()
    return


@app.cell
def _(mo):
    mo.md(r"""
    You can also reload a cached job from a previous session:

    ```python
    from monarch.job.spmd import job_load

    job = job_load(".monarch/job_state.pkl")
    job.run_spmd()  # runs on same reserved hosts
    ```

    ### Remote Debugging

    Add `breakpoint()` anywhere in your training script, then attach from a
    separate terminal:

    ```bash
    $ monarch debug
    ```

    This opens an interactive pdb session across all ranks.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Multi-Node Training with SLURM

    The examples above use `scheduler="local_cwd"` for single-node training. When
    running on a cluster with a scheduler like SLURM, swap the command list for a
    `torchx` `AppDef` — the same `serve()` + `run_spmd()` pattern applies:

    ```python
    from torchx import specs
    from monarch.job.spmd import serve

    app = specs.AppDef(
        name="multi-node-training",
        roles=[
            specs.Role(
                name="trainer",
                image="\",
                entrypoint="torchrun",
                args=[
                    "--nnodes=2",
                    "--nproc-per-node=8",
                    "--rdzv-backend=c10d",
                    "--rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT",
                    TRAIN_SCRIPT,
                ],
                num_replicas=2,
                resource=specs.Resource(cpu=32, gpu=8, memMB=256000),
            ),
        ],
    )

    job = serve(app, scheduler="slurm", scheduler_cfg={"partition": "gpu"})
    job.run_spmd()

    # Edit train.py, then re-run without reprovisioning:
    job.run_spmd()
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
