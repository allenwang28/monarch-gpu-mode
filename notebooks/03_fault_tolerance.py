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
    import random
    from monarch.actor import Actor, endpoint, current_rank, MeshFailure, this_host

    return Actor, MeshFailure, endpoint, this_host


@app.cell
def _(mo):
    mo.md(r"""
    # Fault Tolerance in Monarch

    In the last notebook, you caught a single error with try/except on a
    FlakyWorker. That works when you know which actor failed and have a backup
    ready. But when failures hit every 3 hours across 16,000 GPUs, you need
    patterns that scale.

    This notebook covers three layers of fault handling

    1. **Try/except with retry** — the 90% solution you'll reach for first
    2. **Supervision trees** — centralized error surfacing for infrastructure builders
    3. **TorchFT** — quorum-based training that continues through failures
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Supervision Tree

    Large scale training will run into issues with faulty hardware, flaky networks, and
    software bugs. Monarch is designed to provide good behaviors for errors and faults by
    default with the option to define more sophisticated behavior for faster recovery or
    fault tolerance.

    We borrow from Erlang the idea of a tree of supervision. Each process, and each actor
    has a parent that launched it and is responsible for its health.
    """)
    return


@app.cell
def _(Actor, endpoint):
    from monarch.actor import this_proc

    class Errorful(Actor):
        @endpoint
        def say_hello(self):
            raise Exception("I don't want to")

    e = this_proc().spawn("errorful", Errorful)
    try:
        e.say_hello.call_one().get()
    except Exception:
        print("It didn't say hello")
    return


@app.cell
def _(mo):
    mo.md(r"""
    If we are calling the endpoint and expecting a response, the error gets routed to
    the caller. But we might also call something and provide it no way to respond — in
    that case the error will be delivered to the owner of the actor:

    ```python
    e.say_hello.broadcast()  # do not expect a response
    ```

    The default behavior of the supervision tree means that an error at any level of process
    or actor creation will not go unreported. This can prevent a lot of accidental deadlocks
    compared to the standard unix-style defaults where threads and processes exiting do not
    stop the spawning processes.

    ### Fine-grained Supervision

    Sometimes fine-grained recovery is possible. For instance, if a data loader failed to
    read a URL, perhaps it would work to just restart it. If an actor defines a `__supervise__`
    special method, then it will get called to handle supervision events for meshes owned by
    the actor. If an error happens on an ActorMesh that is a reference, such as a slice, or a
    mesh that is sent to another actor, then the recovery is done on the original creator of
    that mesh, not the holder of the reference.
    """)
    return


@app.cell
def _(Actor, MeshFailure):
    import monarch.actor

    class SupervisorActor(Actor):
        def __supervise__(self, failure: MeshFailure):
            print(f"Failure received: {failure}")
            # If a truthy value is returned, the supervision event is considered handled
            # and will not be propagated further.
            # Otherwise, the event will be propagated to the creator of this actor.
            return True

    return


@app.cell
def _(mo):
    mo.md(r"""
    If a `MeshFailure` is not handled by any `__supervise__` in the supervision tree, it will
    reach the client, where `monarch.actor.unhandled_fault_hook` will be called with the
    `MeshFailure` object. By default, this function crashes the client process with exit code 1.
    """)
    return


@app.cell
def _(MeshFailure):
    import monarch.actor as _monarch_actor

    def my_unhandled_fault_hook(failure: MeshFailure) -> None:
        print(f"Mesh failure was not handled: {failure}")

    _monarch_actor.unhandled_fault_hook = my_unhandled_fault_hook
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Supervision Trees: Centralized Error Surfacing

    Try/except works when the **caller** catches failures. But what about failures
    you don't directly trigger — a child actor crashing in the background, a
    cascading failure deep in a hierarchy?

    Monarch's answer is the __supervise__ API. When an actor spawns children, it becomes
    their supervisor. If a child fails, the supervisor's `__supervise__` method is
    called with a `MeshFailure` describing what went wrong.

    Think of it like a management chain: when an employee hits a problem they can't
    solve, it escalates to their manager. The manager can handle it (return `True`)
    or escalate further (return `False`).
    """)
    return


@app.cell
async def _(Actor, MeshFailure, endpoint, this_host):
    import asyncio

    class ActorCrash(BaseException):
        pass

    class FragileWorker(Actor):
        """A worker that crashes when asked."""

        @endpoint
        async def do_work(self) -> str:
            raise ActorCrash("GPU memory error on this worker!")

    class Supervisor(Actor):
        """Spawns and supervises children. Catches their failures automatically."""

        def __init__(self, child_procs):
            # Supervisor owns these children because it calls spawn()
            self.workers = child_procs.spawn("fragile", FragileWorker)
            self.failure_log = []
            self._supervised = asyncio.Event()

        def __supervise__(self, failure: MeshFailure) -> bool:
            """Called automatically when an owned child actor fails."""
            self.failure_log.append(str(failure))
            print(f"[Supervisor] Caught failure: {failure}")
            self._supervised.set()
            return True  # Handled — don't propagate further

        @endpoint
        async def crash_a_child(self):
            """Tell the child to crash, then wait for supervision to fire."""
            self.workers.do_work.broadcast()  # fire-and-forget — triggers supervision
            await asyncio.wait_for(self._supervised.wait(), timeout=30)

        @endpoint
        async def get_failure_log(self) -> list:
            """Return failures caught by __supervise__ (as strings)."""
            return list(self.failure_log)

    # Create two separate proc meshes (supervisor and children MUST be on different meshes)
    supervisor_procs = this_host().spawn_procs(per_host={"gpus": 1})
    child_procs = this_host().spawn_procs(per_host={"gpus": 1})

    # Spawn supervisor, passing child procs for it to own
    supervisor = supervisor_procs.spawn("supervisor", Supervisor, child_procs)

    # Tell supervisor to trigger the child — broadcast() means the error has
    # no caller to return to, so the actor crashes and supervision fires
    await supervisor.crash_a_child.call_one()

    # Query the supervisor's failure log
    failures = await supervisor.get_failure_log.call_one()
    print(f"\n=== Supervisor's failure log ({len(failures)} entries) ===")
    for j, f in enumerate(failures):
        print(f"  [{j}] {f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What Just Happened

    The supervisor **owns** the `FragileWorker` because it called `spawn()`. When
    the child raised `ActorCrash` (a `BaseException`), the actor died and Monarch
    delivered a `MeshFailure` to the supervisor's `__supervise__` method.


    Key points:

    - `__supervise__` receives a `MeshFailure`
    - Return `True` -> "I handled it" (stops propagation up the tree)
    - Return `False` -> escalate to the supervisor's supervisor
    - If no one handles it, the failure reaches the client as an unhandled fault

    This allows actors to define policies of what to do if Actors they own fail.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### From One Supervisor to a Tree

    You just saw a single supervisor catch one child's failure. Now imagine this
    at scale — supervision forms a **tree** where failures propagate upward:

    ```
    Your Script (root)
    └── Orchestrator
        ├── TrainerSupervisor
        │   ├── Trainer Rank 0  ✓
        │   ├── Trainer Rank 1  ✓
        │   └── Trainer Rank 2  ✗ OOM
        └── GeneratorSupervisor
            ├── Generator 0     ✓
            └── Generator 1     ✓
    ```

    When Trainer Rank 2 crashes, `TrainerSupervisor.__supervise__` fires first.
    If it returns `True`, the failure is handled locally — the Orchestrator never
    even knows. If it returns `False`, the TrainerSupervisor itself fails, and the
    event propagates up to the Orchestrator.

    This is just like exception handling in a call stack:

    ```python
    try:                          # Orchestrator.__supervise__
        try:                      # TrainerSupervisor.__supervise__
            raise OOMError()      # Trainer Rank 2 crashes
        except OOMError:          #   -> handle or re-raise
            handle_or_reraise()
    except Exception:             #   -> Orchestrator handles or crashes
        handle_or_crash()
    ```

    The payoff: instead of grepping through 16,000 log files, you get **one
    structured chain** showing exactly what failed and how the failure propagated.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## TorchFT: Fault-Tolerant Training at Scale

    Recall the 419 interruptions from Notebook 1 — at that scale, restarting
    the entire job each time is unacceptable. You need training to **continue
    with healthy replicas** while failed ones recover.

    The result: **60% faster recovery** than full SLURM restarts, with process
    failures recovering in **~90 seconds**. Here's how.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### How TorchFT Works

    TorchFT provides quorum-based training — instead of stopping the whole job
    when a replica fails, the remaining replicas continue:

    ```
    Traditional Training:
    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
    │Replica 0│ │Replica 1│ │Replica 2│ │Replica 3│
    └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
         │           │           │           │
         └───────────┴─────┬─────┴───────────┘
                           ▼
                  All-reduce every step
                  Single failure → restart everything

    TorchFT Quorum-Based Training:
    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
    │Replica 0│ │Replica 1│ │Replica 2│ │Replica 3│
    │ healthy │ │  FAILED │ │ healthy │ │ healthy │
    └────┬────┘ └─────────┘ └────┬────┘ └────┬────┘
         │                       │           │
         └───────────┬───────────┴───────────┘
                     ▼
            Quorum shrinks to 3 replicas
            Training continues without stopping
            Failed replica recovers and rejoins
    ```

    On failure, TorchFT catches collective errors through its own
    ManagedProcessGroup, isolates the failure to a single replica group, and
    continues training without restarting the entire job.

    A Lighthouse service (Rust gRPC) coordinates quorum membership via
    heartbeats (~100ms), while a Manager handles checkpoint recovery when
    replicas join or leave.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Monarch + TorchFT: Separation of Concerns

    TorchFT handles **step-level fault tolerance** — catching errors mid-training
    and shrinking the quorum. Monarch handles **orchestration-level recovery** —
    respawning crashed replicas, managing job allocation, and escalating from
    fast process restarts to full reallocation when needed.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Case Study: Qwen3-32B Training with Fault Injection

    We ran this on a **30 node (240 H100s) Coreweave cluster**, training
    Qwen3-32B using torchtitan and TorchFT.

    **Test conditions:**

    - Failures injected at ~3-minute intervals, 100 total events
    - Multiple failure modes: segfaults, process kills, NCCL abort, host
      eviction, GIL deadlock

    **Results:**

    | Metric | Result |
    |--------|--------|
    | Recovery speed vs full SLURM restart | **60% faster** |
    | Avg recovery time (process failures) | **90 seconds** |
    | Avg recovery time (machine failures) | **2.5 minutes** |

    **Why the improvement?**
    Monarch allows for configurable recovery strategies based on failure type —
    we avoid unnecessary job rescheduling by attempting fast process-level
    restarts first.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    For more details on error handling in Monarch, see the
    [Error Handling in Meshes](https://meta-pytorch.org/monarch/actors.html#error-handling-in-meshes)
    documentation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    **Key takeaways:**

    1. **Try/except first**: The canonical pattern covers 90% of cases
    2. **Supervision trees**: Centralized error surfacing — one place to see
       what went wrong
    3. **Quorum-based training**: Continue with healthy replicas while failed
       ones recover
    4. **Monarch + TorchFT**: Orchestration-level recovery + step-level fault
       tolerance

    These three patterns compose — try/except is your first line of defense,
    supervision gives you visibility, and TorchFT handles training-specific
    recovery at scale.

    We've built patterns to handle failures. Next: putting it all together —
    services, weight sync, and async RL loops.
    """)
    return


if __name__ == "__main__":
    app.run()
