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
    # Services: Building a Generator Pool

    We just introduced async RL and why it matters. If you've used Claude Code or any coding-capable AI, you know that depending on the difficulty of your question, it could take a really long time to generate even one trajectory. As we showed in the previous notebook, there's also significant variance in how long generation takes.

    For this reason, you'll typically want to run RL workloads with **many generator replicas running in parallel**.

    ## The "Service" Abstraction

    When you get to this point, there are a few ways you could model this. A pretty natural approach is to create a **"service"**.

    In traditional software engineering, a service is a component that:
    - Handles requests from clients
    - Can be replicated for throughput and reliability
    - Lives behind some kind of load balancer (think Kubernetes)

    This idea applies here too. You *could* find and grab an off-the-shelf implementation, but since we're always working in Python, it would be nice to have a **Python-native implementation** - especially one you can easily customize to match your needs as an RL researcher.

    ## What We'll Build

    There are a lot of ideas that could go into a service, but what we'll focus on implementing here - using **pure Monarch** - is a service that can:

    1. **Load balance** across replicas (round-robin for simplicity, but you could implement more exotic strategies based on your workload)
    2. **Track and respond to faults** (health tracking, retry with different replica)
    3. **Be discoverable** (services register themselves, clients find them by name)
    4. **Support multi-GPU replicas** (each replica is an ActorMesh that can span multiple GPUs)
    5. **Target specific replicas** (grab a replica and keep using it - useful for stateful interactions)

    Since each replica is an ActorMesh, everything we build here *could* run multi-host - the abstraction supports it. For this tutorial, we'll run on a single node to keep things simple.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Quick Recap: Monarch Actors

    Before we dive in, a quick refresher on the Monarch primitives we'll use. If you've gone through the earlier notebooks, you'll recognize these:

    - **`@endpoint`** - Marks a method as callable from other actors. Returns a `Future` that you `.get()` to await.
    - **`__supervise__`** - Called when a child actor fails. You can log the error and decide whether to handle it.
    - **`get_or_spawn_controller`** - Gets or creates a singleton actor by name. First caller spawns it, subsequent callers get the existing one.
    - **`ActorMesh`** - A group of actors spawned together. When you call an endpoint, it runs on all actors in the mesh.

    With that in mind, let's think about what API we want for our service.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A Key Assumption: 1 Replica = 1 ActorMesh

    Most workloads we care about will need to span multiple GPUs. A 70B model needs tensor parallelism across 8 GPUs. Models like DeepSeek V3 may need to span multiple *hosts*.

    So rather than treating single-GPU replicas as the default and multi-GPU as a special case, we'll make a basic assumption: **1 replica = 1 ActorMesh**.

    This means:
    - Each replica is a group of workers (an ActorMesh)
    - Workers within a replica coordinate via **collectives** - each replica is its own distributed "world"
    - The service manages replicas, not individual workers

    This design composes naturally with the existing ML ecosystem. A common pattern might be to plug **vLLM** or another inference engine into one of these meshes - it just sees a standard distributed environment.

    If you happen to have a small model that fits on one GPU, you just have a replica with 1 worker. The abstraction still works.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Designing the API First

    Before we implement anything, let's sketch what we want the usage to look like. This helps us work backwards to the implementation.

    **What a client wants to do:**
    ```python
    # Get a service by name
    gen_service = get_service("generators")

    # Get a replica and call it
    replica, replica_idx = gen_service.get_replica_with_idx()
    result = replica.generate(task)

    # If something fails, mark it unhealthy and retry
    gen_service.mark_unhealthy(replica_idx)
    ```

    **What a service wants to do on startup:**
    ```python
    # Register itself so clients can find it
    register_service("generators", self)
    ```

    This gives us a clean separation: services register themselves, clients discover them by name. You *can* pass actor references around directly - that works fine - but referring to services by name is often a cleaner pattern, especially when services and clients are spawned independently.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Service Discovery: The Registry Pattern

    Let's implement the discovery mechanism first. We'll create a `ServiceRegistry` singleton and wrap it with helper functions.
    """)
    return


@app.cell
def _():
    from monarch.actor import Actor, endpoint, get_or_spawn_controller

    class ServiceRegistry(Actor):
        """
        Singleton registry for service discovery.

        Found via get_or_spawn_controller("service_registry", ServiceRegistry).
        First caller spawns it, subsequent callers get the existing one.
        """

        def __init__(self):
            self.services: dict[str, Actor] = {}
            print("[REGISTRY] ServiceRegistry spawned")

        @endpoint
        def register(self, name: str, service) -> None:
            """Register a service by name."""
            self.services[name] = service
            print(f"[REGISTRY] Registered '{name}'")

        @endpoint
        def get(self, name: str):
            """Get a service by name."""
            if name not in self.services:
                raise KeyError(f"Service '{name}' not found")
            return self.services[name]

        @endpoint
        def list_services(self) -> list[str]:
            """List all registered service names."""
            return list(self.services.keys())


    def _get_registry():
        """Get the singleton ServiceRegistry."""
        return get_or_spawn_controller("service_registry", ServiceRegistry).get()


    def get_service(name: str):
        """Get a service by name. Raises KeyError if not found."""
        registry = _get_registry()
        return registry.get.call_one(name).get()


    def register_service(name: str, service) -> None:
        """Register a service so others can find it by name."""
        registry = _get_registry()
        registry.register.call_one(name, service).get()


    def list_services() -> list[str]:
        """List all registered service names."""
        registry = _get_registry()
        return registry.list_services.call_one().get()
    return Actor, endpoint, get_service, register_service


@app.cell
def _(mo):
    mo.md(r"""
    Now clients can simply call:
    ```python
    gen_service = get_service("generators")
    ```

    And services register themselves with:
    ```python
    register_service("generators", self)
    ```

    The registry is a singleton - `get_or_spawn_controller` ensures everyone gets the same one.

    ## The Worker: Where the Actual Work Happens

    Let's define a worker actor. Each worker knows:
    - Its **replica ID** - which replica it belongs to (passed at spawn time)
    - Its **local rank** - its position within the replica's ActorMesh (0, 1, 2, ...)
    """)
    return


@app.cell
def _(Actor, endpoint):
    from monarch.actor import current_rank

    class Worker(Actor):
        """A worker that does work. Knows its replica ID and local rank."""

        def __init__(self, replica_id: int, fail_rate: float = 0.0):
            self.replica_id = replica_id
            # Local rank within this replica's ActorMesh
            self.local_rank = current_rank().rank  # 0, 1, 2, ... within this replica
            self.fail_rate = fail_rate
            self.calls = 0
            print(f"[WORKER] Replica {replica_id}, local rank {self.local_rank} ready")

        @endpoint
        def ping(self) -> bool:
            """Health check endpoint. Returns True if the worker is alive."""
            return True

        @endpoint
        def process(self, data: str) -> dict:
            """Process some data. Might fail based on fail_rate."""
            import random
            self.calls += 1

            if random.random() < self.fail_rate:
                raise RuntimeError(f"Worker failed! (replica={self.replica_id}, local={self.local_rank})")

            return {
                "replica_id": self.replica_id,
                "local_rank": self.local_rank,
                "calls": self.calls,
                "result": f"Processed '{data}' by replica {self.replica_id}",
            }

        @endpoint
        def get_replica_id(self) -> int:
            """Return which replica this worker belongs to."""
            return self.replica_id

        @endpoint
        def get_local_id(self) -> int:
            """Return this worker's local rank within its replica."""
            return self.local_rank
    return (Worker,)


@app.cell
def _(mo):
    mo.md(r"""
    The **local rank** is the worker's position within its ActorMesh (0, 1, 2, ...). This is how workers within a replica coordinate - for example, rank 0 might handle the first shard of a tensor-parallel model.

    The **replica ID** is passed at spawn time so workers know which replica they belong to. This is useful for logging and debugging.

    ## The Service: Routing and Health Tracking

    Now let's build the service that manages replicas. It needs to:
    - Slice a ProcMesh into replica-sized chunks
    - Spawn an ActorMesh for each replica (passing the replica ID)
    - Route requests round-robin to healthy replicas
    - Track which replicas are healthy
    - Handle failures via `__supervise__`
    """)
    return


@app.cell
def _(Actor, endpoint):
    class Service(Actor):
        """
        Manages a pool of worker replicas with routing and health tracking.

        Each replica is an ActorMesh (potentially spanning multiple GPUs/hosts).
        The service:
        1. Slices resources into replica-sized chunks
        2. Spawns an ActorMesh per replica
        3. Routes requests round-robin to healthy replicas
        4. Tracks health and handles failures
        5. Can probe unhealthy replicas to check if they've recovered
        """

        def __init__(
            self,
            service_name: str,
            worker_class: type,
            replica_procs,
            procs_per_replica: int,
            **worker_kwargs
        ):
            self.service_name = service_name
            total_procs = len(replica_procs)
            self.num_replicas = total_procs // procs_per_replica
            self.procs_per_replica = procs_per_replica

            # Slice the proc mesh into replica-sized chunks
            self.replicas = []
            for i in range(self.num_replicas):
                start = i * procs_per_replica
                end = start + procs_per_replica

                # Slice out this replica's procs
                replica_slice = replica_procs.slice(procs=slice(start, end))

                # Spawn an ActorMesh on this slice, passing replica_id
                replica_mesh = replica_slice.spawn(
                    f"replica_{i}",
                    worker_class,
                    replica_id=i,  # Pass replica ID to workers
                    **worker_kwargs
                )
                self.replicas.append(replica_mesh)

            # Track health: healthy and unhealthy sets
            self.healthy = set(range(self.num_replicas))
            self.unhealthy: set[int] = set()

            # Round-robin index
            self.next_idx = 0

            print(f"[SERVICE:{service_name}] {self.num_replicas} replicas × {procs_per_replica} procs each")

        def __supervise__(self, failure) -> bool:
            """Called when a worker fails. Log it."""
            report = failure.report()
            print(f"[SERVICE:{self.service_name}] Failure detected: {report[:100]}...")
            return True  # Handled - real recovery is caller retrying with different replica

        @endpoint
        def get_replica(self):
            """Get a healthy replica (round-robin selection)."""
            if not self.healthy:
                raise RuntimeError("No healthy replicas available!")

            healthy_list = sorted(self.healthy)
            idx = self.next_idx % len(healthy_list)
            self.next_idx += 1

            return self.replicas[healthy_list[idx]]

        @endpoint
        def get_replica_with_idx(self):
            """Get a healthy replica and its index (for marking unhealthy later)."""
            if not self.healthy:
                raise RuntimeError("No healthy replicas available!")

            healthy_list = sorted(self.healthy)
            idx = self.next_idx % len(healthy_list)
            self.next_idx += 1

            replica_idx = healthy_list[idx]
            return self.replicas[replica_idx], replica_idx

        @endpoint
        def mark_unhealthy(self, replica_idx: int) -> None:
            """Mark a replica as unhealthy."""
            if replica_idx in self.healthy:
                self.healthy.discard(replica_idx)
                self.unhealthy.add(replica_idx)
                print(f"[SERVICE:{self.service_name}] Replica {replica_idx} marked unhealthy. "
                      f"Healthy: {len(self.healthy)}/{self.num_replicas}")

        @endpoint
        def mark_healthy(self, replica_idx: int) -> None:
            """Mark a replica as healthy again."""
            if replica_idx < self.num_replicas:
                self.unhealthy.discard(replica_idx)
                self.healthy.add(replica_idx)

        @endpoint
        def check_health(self) -> dict:
            """
            Probe all unhealthy replicas to see if they've recovered.
            Returns a dict with recovery results.
            """
            recovered = []
            still_unhealthy = []

            for replica_idx in list(self.unhealthy):
                try:
                    # Ping the replica to see if it's alive
                    self.replicas[replica_idx].ping.call_one().get()
                    # Success - mark healthy
                    self.unhealthy.discard(replica_idx)
                    self.healthy.add(replica_idx)
                    recovered.append(replica_idx)
                    print(f"[SERVICE:{self.service_name}] Replica {replica_idx} recovered!")
                except Exception:
                    # Still unhealthy
                    still_unhealthy.append(replica_idx)

            return {
                "recovered": recovered,
                "still_unhealthy": still_unhealthy,
                "healthy_count": len(self.healthy),
            }

        @endpoint
        def get_health_status(self) -> dict:
            return {
                "total": self.num_replicas,
                "healthy": len(self.healthy),
                "unhealthy": len(self.unhealthy),
                "healthy_indices": sorted(self.healthy),
                "unhealthy_indices": sorted(self.unhealthy),
            }
    return (Service,)


@app.cell
def _(mo):
    mo.md(r"""
    Key design decisions:

    1. **1 replica = 1 ActorMesh** - Each replica can span multiple GPUs (or even hosts)
    2. **Pass replica_id at spawn** - Workers know which replica they belong to
    3. **Caller registers the service** - After spawning, the caller calls `register_service()`. We can't do this in `__init__` because actors can't block on Futures during initialization.
    4. **Round-robin routing** - Simple, predictable, works well when replicas are similar
    5. **Caller marks unhealthy** - The service doesn't automatically detect which replica failed. Instead, the caller tells it after catching an exception.
    6. **Health probes for recovery** - Workers have a `ping()` endpoint. The service's `check_health()` method probes unhealthy replicas and recovers them if they respond.
    7. **`__supervise__`** - Monarch calls this when a child actor fails. We log it, but the real handling is the caller retrying with a different replica.

    ## Let's Run It!

    Let's create a simple demo that shows:
    - Different replicas being called (round-robin)
    - What happens when a replica fails
    - How the service routes around failures
    - How health checks recover replicas
    """)
    return


@app.cell
def _(Service, Worker, register_service):
    # Demo: Create a service with 4 replicas, 1 worker each
    # We'll send 8 requests to see round-robin in action

    from monarch.actor import this_host

    # Get procs on this host
    host = this_host()

    # The Service itself is an actor, so it needs to be spawned on a proc
    service_proc = host.spawn_procs({"procs": 1})

    # Procs for the worker replicas
    worker_procs = host.spawn_procs({"procs": 4})

    # Spawn the service actor
    demo_service = service_proc.spawn(
        "demo_service",
        Service,
        service_name="demo",
        worker_class=Worker,
        replica_procs=worker_procs,
        procs_per_replica=1,
        fail_rate=0.0,  # No failures for now
    )

    # Register the service so it can be discovered by name
    register_service("demo", demo_service)

    # Send 8 requests - should round-robin through replicas 0,1,2,3,0,1,2,3
    print("Round-robin demo:")
    for _i in range(8):
        replica, replica_idx = demo_service.get_replica_with_idx.call_one().get()
        result = replica.process.call_one(f"request_{_i}").get()
        print(f"  Request {_i}: handled by replica {result['replica_id']}")

    # Note: In production you'd call worker_procs.stop() and service_proc.stop()
    # to clean up, but there's currently a Monarch bug where stopping procs
    # can crash the kernel if there are lingering references.
    return (host,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Handling Failures

    Now let's see what happens when a replica fails. We'll create a service with a 30% failure rate and watch the retry logic in action. Then we'll use `check_health()` to recover replicas.
    """)
    return


@app.cell
def _(Service, Worker, host, register_service):
    # Create a service with a failure rate
    # Service needs its own proc, workers need their procs
    flaky_service_proc = host.spawn_procs({"procs": 1})
    flaky_worker_procs = host.spawn_procs({"procs": 4})

    flaky_service = flaky_service_proc.spawn(
        "flaky_service",
        Service,
        service_name="flaky_demo",
        worker_class=Worker,
        replica_procs=flaky_worker_procs,
        procs_per_replica=1,
        fail_rate=0.3,  # 30% failure rate
    )

    # Register the service
    register_service("flaky_demo", flaky_service)

    # Try to process with retry
    def process_with_retry(svc, data, max_retries=3):
        for attempt in range(max_retries):
            replica, replica_idx = svc.get_replica_with_idx.call_one().get()
            try:
                result = replica.process.call_one(data).get()
                print(f"  Success on replica {replica_idx}")
                return result
            except Exception as e:
                print(f"  Attempt {attempt+1} failed on replica {replica_idx}: {e}")
                svc.mark_unhealthy.call_one(replica_idx).get()
        raise RuntimeError("All retries exhausted")

    # Process several requests
    print("Failure handling demo:")
    for _j in range(5):
        print(f"Request {_j}:")
        try:
            process_with_retry(flaky_service, f"data_{_j}")
        except RuntimeError as e:
            print(f"  {e}")

    # Check health status after failures
    status = flaky_service.get_health_status.call_one().get()
    print(f"\nAfter failures: {status['healthy']}/{status['total']} replicas healthy")
    print(f"Unhealthy replicas: {status['unhealthy_indices']}")

    # Now run health checks to recover replicas
    # (In our case, the replicas are still alive - they just had random failures)
    print("\nRunning health check to recover replicas...")
    recovery = flaky_service.check_health.call_one().get()
    print(f"Recovered: {recovery['recovered']}")
    print(f"Still unhealthy: {recovery['still_unhealthy']}")

    # Check health status after recovery
    status_after = flaky_service.get_health_status.call_one().get()
    print(f"\nAfter health check: {status_after['healthy']}/{status_after['total']} replicas healthy")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Real Example: Distributed Zorplex with Transformers

    Now let's build something real. We'll create:
    - A **Generator Service** with multiple replicas, each with its own GPU
    - A **Zorplex Service** that evaluates generated trajectories
    - A client that distributes prompts across generators

    Each generator replica will:
    - Have its own GPU (based on replica_id)
    - Load the model once at startup
    - Process prompts as they come in
    """)
    return


app._unparsable_cell(
    """
    # Generator Worker - each replica loads a model on its assigned GPU
    import torch

    class GeneratorWorker(Actor):
        \"\"\"
        A generator worker that loads a model and generates trajectories.
        Each replica gets its own GPU.
        \"\"\"

        def __init__(self, replica_id: int, model_name: str = \"Qwen/Qwen2.5-0.5B-Instruct\"):
            self.replica_id = replica_id
            self.local_rank = 0  # Single-GPU replicas for this example

            # Each replica uses a different GPU
            self.device = f\"cuda:{replica_id}\" if torch.cuda.is_available() else \"cpu\"

            print(f\"[GEN:{replica_id}] Loading {model_name} on {self.device}...\")

            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device)

            print(f\"[GEN:{replica_id}] Ready!\")

        @endpoint
        def generate(self, prompt: str, max_new_tokens: int = 100) -> dict:
            \"\"\"Generate a response for a prompt.\"\"\"
            print(f\"[GEN:{self.replica_id}]} Generating based on prompt {prompt}\")
            inputs = self.tokenizer(prompt, return_tensors=\"pt\").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {
                \"replica_id\": self.replica_id,
                \"prompt\": prompt,
                \"response\": response,
            }

        @endpoint
        def get_replica_id(self) -> int:
            return self.replica_id
    """,
    name="_"
)


@app.cell
def _(Actor, endpoint):
    # Zorplex Evaluator Service

    class ZorplexWorker(Actor):
        """
        Evaluates zorplex task responses.
        Stateless - can run on CPU.
        """

        def __init__(self, replica_id: int):
            self.replica_id = replica_id
            print(f"[ZORPLEX:{replica_id}] Ready!")

        @endpoint
        def evaluate(self, task_type: str, question: str, correct_answer, response: str) -> dict:
            """Evaluate a response against a zorplex task."""
            from zorplex_rl import get_spec
            from zorplex_rl.task_specs import Task

            spec = get_spec(task_type)
            # Reconstruct the Task object
            task = Task(question=question, correct_answer=correct_answer)
            eval_result = spec.evaluate(task, response)

            return {
                "replica_id": self.replica_id,
                "task_type": task_type,
                "is_correct": eval_result.is_correct,
                "response": response[:100] + "..." if len(response) > 100 else response,
            }

        @endpoint
        def get_replica_id(self) -> int:
            return self.replica_id
    return (ZorplexWorker,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Putting It Together

    Now we wire up the full distributed system with generators and evaluators.
    """)
    return


@app.cell
def _(
    GeneratorWorker,
    Service,
    ZorplexWorker,
    get_service,
    register_service,
    torch,
):
    from monarch.actor import this_host as _this_host

    # Get a fresh host for this demo
    _host = _this_host()

    # Allocate resources - limit to 4 replicas max
    num_gpus = min(torch.cuda.device_count() if torch.cuda.is_available() else 2, 4)
    print(f"Using {num_gpus} generator replicas")

    # Service actors need their own procs
    gen_service_proc = _host.spawn_procs({"procs": 1})
    eval_service_proc = _host.spawn_procs({"procs": 1})

    # Worker procs
    gen_worker_procs = _host.spawn_procs({"procs": num_gpus})
    eval_worker_procs = _host.spawn_procs({"procs": 2})

    # Create generator service
    gen_service = gen_service_proc.spawn(
        "gen_service",
        Service,
        service_name="generators",
        worker_class=GeneratorWorker,
        replica_procs=gen_worker_procs,
        procs_per_replica=1,
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
    )

    # Wait for generator service to be fully initialized (model loading takes time)
    gen_status = gen_service.get_health_status.call_one().get()
    print(f"Generator service ready: {gen_status['healthy']}/{gen_status['total']} replicas")

    # Create evaluator service
    eval_service = eval_service_proc.spawn(
        "eval_service",
        Service,
        service_name="evaluators",
        worker_class=ZorplexWorker,
        replica_procs=eval_worker_procs,
        procs_per_replica=1,
    )

    # Wait for evaluator service to be fully initialized
    eval_status = eval_service.get_health_status.call_one().get()
    print(f"Evaluator service ready: {eval_status['healthy']}/{eval_status['total']} replicas")

    # Register services so they can be discovered
    register_service("generators", gen_service)
    register_service("evaluators", eval_service)

    # Generate a pool of zorplex tasks
    from zorplex_rl import get_spec, TASK_SPECS

    tasks = []
    for task_type in list(TASK_SPECS.keys())[:3]:  # Just first 3 task types for demo
        spec = get_spec(task_type)
        for _ in range(2):
            task = spec.generate_task()  # Returns a Task dataclass
            tasks.append({
                "task_type": task_type,
                "question": task.question,
                "correct_answer": task.correct_answer,
            })

    print(f"Generated {len(tasks)} tasks across {len(set(t['task_type'] for t in tasks))} task types")

    # Distribute across generators
    results = []
    for task in tasks:
        # Get a generator replica
        gen_replica, gen_idx = get_service("generators").get_replica_with_idx.call_one().get()

        # Generate
        gen_result = gen_replica.generate.call_one(task["question"]).get()
        print(f"Task '{task['task_type']}' -> Generator {gen_result['replica_id']}")

        # Get an evaluator replica
        eval_replica, eval_idx = get_service("evaluators").get_replica_with_idx.call_one().get()

        # Evaluate - pass individual fields since Task isn't serializable
        eval_result = eval_replica.evaluate.call_one(
            task["task_type"],
            task["question"],
            task["correct_answer"],
            gen_result["response"]
        ).get()
        results.append(eval_result)

    # Summary
    correct = sum(1 for r in results if r["is_correct"])
    print(f"\nResults: {correct}/{len(results)} correct")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What We Built

    A simple but complete service framework:

    | Component | Responsibility |
    |-----------|---------------|
    | **`get_service(name)`** | Find a service by name |
    | **`register_service(name, svc)`** | Register a service for discovery |
    | **Worker** | Does the actual work, knows its replica_id and local_rank |
    | **Service** | Owns replicas (ActorMeshes), routes requests, tracks health |

    The pattern:
    1. Services register themselves on startup with `register_service()`
    2. Clients discover services with `get_service()`
    3. Clients get replicas, make calls, handle failures inline
    4. On failure, mark unhealthy and retry with a different replica

    ## What's Missing?

    The generators need **updated weights** after the trainer takes a gradient step. Right now they'd be using stale weights forever.

    How do we efficiently sync weights from trainer to generators?

    **Next notebook: RDMA Weight Sync** - Using Monarch's RDMA primitives to transfer weights without blocking training.
    """)
    return


if __name__ == "__main__":
    app.run()
