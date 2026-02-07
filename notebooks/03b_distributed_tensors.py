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
    # Distributed Tensors in Monarch

    Monarch can broadcast tensor compute to a mesh of processes, allowing a single
    controller to do distributed tensor compute.
    """)
    return


@app.cell
def _():
    import monarch
    import torch
    import torch.nn as nn
    from monarch.actor import this_host

    torch.set_default_device("cuda")
    return monarch, nn, this_host, torch


@app.cell
def _(mo):
    mo.md(r"""
    ## Meshes

    All computation is done on a 'mesh' of devices.
    Here we create a mesh composed of the machine running the notebook:
    """)
    return


@app.cell
def _(this_host):
    mesh = this_host().spawn_procs({"gpu": 8})
    print(mesh.to_table())
    return (mesh,)


@app.cell
def _(mo):
    mo.md(r"""
    Without a mesh active, torch runs locally.
    """)
    return


@app.cell
def _(torch):
    torch.rand(3, 4)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Once active, torch runs on every device in the mesh.
    """)
    return


@app.cell
def _(mesh, torch):
    with mesh.activate():
        t = torch.rand(3, 4, device="cuda")
    t
    return (t,)


@app.cell
def _(mo):
    mo.md(r"""
    `inspect` moves rank 0's copy of `t` to the notebook for debugging.
    Providing coordinates lets us inspect other ranks' copies.
    """)
    return


@app.cell
def _(monarch, t):
    monarch.inspect(t)
    monarch.show(t)
    monarch.show(t, gpu=1)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Tensor Commands

    Any command done on the controller, such as multiplying these tensors,
    performs that action to all of the tensors in the collection.
    """)
    return


@app.cell
def _(mesh, monarch, t):
    with mesh.activate():
        obj = t @ t.T
        monarch.show(obj)
    return


@app.cell
def _(mo):
    mo.md(r"""
    If a command fails, the workers stay alive and can execute future commands:
    """)
    return


@app.cell
def _(mesh, monarch, t, torch):
    try:
        with mesh.activate():
            big_w = torch.rand(4, 1024 * 1024 * 1024 * 1024 * 8, device="cuda")
            v = t @ big_w
            monarch.show(v)
    except Exception:
        import traceback
        traceback.print_exc()

    print("RECOVERED!")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Since monarch recovers from errors, you can search for what works:
    """)
    return


@app.cell
def _(mesh, monarch, torch):
    N = 1
    while True:
        try:
            with mesh.activate():
                batch = torch.rand(N, 1024 * 1024 * 1024, device="cuda")
            monarch.inspect(batch.sum())
            N = 2 * N
            print(f"at least 2**{N} elements work")
        except Exception:
            print(f"max is 2**{N} elements")
            break
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Collectives

    Each machine has its own copy of the tensor, similar to `torch.distributed`.

    To compute across tensors in the mesh, we use special communication operators,
    analogous to collectives.
    """)
    return


@app.cell
def _(mesh, monarch, torch):
    with mesh.activate():
        a = torch.rand(3, 4, device="cuda")
        r = a.reduce("gpu", "sum")

    monarch.show(a, gpu=0)
    monarch.show(a, gpu=1)

    monarch.show(r, gpu=0)
    monarch.show(r, gpu=1)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Remote GPUs

    We can also connect to remote GPUs reserved from some scheduler.
    Here we simulate a multi-host setup locally:
    """)
    return


@app.cell
def _(this_host, torch):
    remote_mesh = this_host().spawn_procs({"host": 4, "gpu": 4})

    print(remote_mesh.to_table())
    with remote_mesh.activate():
        eg = torch.rand(3, 4, device="cuda")
        rgpu = eg.reduce("gpu", "sum")
        rhost = eg.reduce("host", "sum")
    return (remote_mesh,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Device Mesh Dimensions

    Meshes can be renamed and reshaped to fit the parallelism desired.
    """)
    return


@app.cell
def _(remote_mesh):
    mesh_2d_parallel = remote_mesh.rename(host="dp", gpu="tp")
    print(mesh_2d_parallel.to_table())

    mesh_3d_parallel = remote_mesh.split(host=("dp", "pp"), gpu=("tp",), pp=2)
    print(mesh_3d_parallel.to_table())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Pipelining

    Pipelining is accomplished by slicing the mesh, and copying tensors from
    one mesh to another.
    """)
    return


@app.cell
def _(remote_mesh):
    pipeline_mesh = remote_mesh.rename(host="pp")
    meshes = [pipeline_mesh.slice(pp=i) for i in range(pipeline_mesh.size("pp"))]
    print(meshes[0].to_table())
    return (meshes,)


@app.cell
def _(mo):
    mo.md(r"""
    Initialize a model across multiple pipeline stages:
    """)
    return


@app.cell
def _(meshes, monarch, nn, torch):
    layers_per_stage = 2
    stages = []
    for stage_mesh in meshes:
        with stage_mesh.activate():
            layers = []
            for _ in range(layers_per_stage):
                layers.extend([nn.Linear(4, 4), nn.ReLU()])
            stages.append(nn.Sequential(*layers))

    def forward_pipeline(x):
        with torch.no_grad():
            for stage_mesh, stage in zip(meshes, stages):
                x = x.to_mesh(stage_mesh)
                with stage_mesh.activate():
                    x = stage(x)
            return x

    with meshes[0].activate():
        input = torch.rand(3, 4, device="cuda")

    output = forward_pipeline(input)
    monarch.show(output)
    print(output.mesh.to_table())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## DDP Example

    The next sections use an example of writing DDP to illustrate a typical way to
    develop code in Monarch.

    We interleave the backward pass with the gradient reductions and parameter updates.
    `monarch.grad_generator` incrementally runs the backward pass, returning an iterator
    that computes the grad parameters one at a time.
    """)
    return


@app.cell
def _(monarch, torch):
    def train(model, input, target):
        loss = model(input, target)
        rparameters = list(reversed(list(model.parameters())))
        grads = monarch.grad_generator(loss, rparameters)
        with torch.no_grad():
            it = iter(zip(rparameters, grads))
            todo = next(it, None)
            while todo is not None:
                param, grad = todo
                grad.reduce_("dp", "sum")
                todo = next(it, None)
                param += 0.01 * grad

    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Simulation

    We can use a simulator to check for expected behavior of code before running it
    for real. It is another kind of mesh, which simulates rather than computes results.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Overlapping Comms/Compute

    Commands on different devices run in parallel, but by default commands on a single
    device run sequentially.

    We introduce parallelism on a device via stream objects. To use a tensor from one
    stream on another we borrow it. The borrow API ensures deterministic memory usage
    and eliminates the race conditions in the `torch.cuda.stream` API.
    """)
    return


@app.cell
def _(monarch):
    main = monarch.get_active_stream()
    comms = monarch.Stream("comms")
    return (comms,)


@app.cell
def _(mo):
    mo.md(r"""
    The DDP example again, but using multiple streams:
    """)
    return


@app.cell
def _(comms, monarch, nn, torch):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            for x in range(8):
                layers.append(nn.Linear(4, 4))
                layers.append(nn.ReLU())
            self.layers = nn.Sequential(*layers)

        def forward(self, input, target):
            output = self.layers(input)
            return torch.nn.functional.cross_entropy(output, target)

    def train2(model, input, target):
        loss = model(input, target)
        rparameters = list(reversed(list(model.parameters())))
        grads = monarch.grad_generator(loss, rparameters)
        with torch.no_grad():
            # NEW: iter also produces the tensor borrowed
            # to the comm stream
            it = iter(
                (param, grad, *comms.borrow(grad, mutable=True))
                for param, grad in zip(rparameters, grads)
            )

            todo = next(it, None)
            while todo is not None:
                param, grad, comm_grad, borrow = todo
                # NEW: compute the reduce on the comm stream
                with comms.activate():
                    comm_grad.reduce_("dp", "sum")
                borrow.drop()
                todo = next(it, None)
                param += 0.01 * grad



    def simulate():
        simulator = monarch.Simulator(hosts=1, gpus=4, trace_mode="stream_only")
        mesh = simulator.mesh.rename(gpu="dp")
        with mesh.activate():
            model = Net()

            train2(model, torch.rand(3, 4), torch.full((3,), 1, dtype=torch.int64))

            try:
                simulator.display()
            except Exception:
                pass

    simulate()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
