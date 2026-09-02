# trimtab

Change inference engine configuration while the engine runs.

A trim tab is the small surface on a ship's rudder that steers the rudder
that steers the ship. trimtab is a small patch and a control plane that steer
big inference engines without restarting them.

## Why

Inference engines freeze their operational parameters at startup. Changing a
queue cap, a batch limit, or a prefill chunk size means killing the process
and reloading weights, which costs minutes per change. Tuning becomes an
offline chore and every experiment pays a reboot tax.

Most of those parameters are plain scheduler state. Nothing on the GPU is
allocated when they change. They are startup flags only because nobody wired
a setter.

## What works today

Two engines, one control surface.

How it works, in plain terms. When SGLang or vLLM starts, it reads its flags
once and copies a few of them into variables on the scheduler object. From
then on the flag is dead and the variable is what the scheduler reads, many
times a second, to decide what to run next. trimtab lets you change that
variable on the running server. Each engine already has a way to send a
command into the scheduler, so the patch adds a handler that takes the new
value, checks it against the ceiling the engine sized memory for at boot, and
writes it. The scheduler picks it up on its next step.

SGLang ships a POST /set_internal_state route wired end to end, from HTTP
through a fan-out communicator to every scheduler rank, guarded by an
allowlist of five niche keys. The patch widens the allowlist and applies the
values to the live scheduler. vLLM dispatches utility RPCs to EngineCore by
method name, so the patch adds one method and one dev router (the server runs
with VLLM_SERVER_DEV_MODE=1, the same gate vLLM puts on its own dev routes).

trimtab supports the knobs listed in the manifests and nothing else. Today
that is three hot knobs on SGLang plus five niche ones upstream already
allowed, two hot knobs on vLLM, and 11 (SGLang) and 7 (vLLM) cold knobs the
supervisor can change by relaunching with new flags. Every other flag is set
at launch as before. A knob becomes hot only after its per-step read is
verified at a source line, which is the last column below. A knob read only at
boot is cold.

| engine | knob | validation | per-step read verified at |
|---|---|---|---|
| sglang | max_running_requests | 1 up to boot value | scheduler.py 2274, 2302, 3626 |
| sglang | max_queued_requests | 0 or more | scheduler.py 3077 |
| sglang | chunked_prefill_size | 1 or more | scheduler.py 3600 |
| vllm | max_num_seqs | 1 up to boot value | v1/core/sched/scheduler.py 792 |
| vllm | max_num_batched_tokens | 1 up to boot value | v1/core/sched/scheduler.py 440, 529 |

Above the engines sits a Python package. Adapters for both engines with one
result shape. A manifest per engine version that classifies every knob hot or
cold and validates at the boundary. An append-only SQLite store where the
version table is the history. A reconciler that converges a replica on the
group's desired version and is a no-op when already there. A CLI. A mock
engine that speaks both control surfaces without a GPU, which is what the
tests and the bench run against on CPU.

Also built. A canary orchestrator that runs a candidate on a subset of
replicas through per-replica overrides, samples Prometheus metrics from both
sides over a window, decides against gates expressed as data, promotes on pass
and reverts on fail. A Prometheus scraper that turns engine /metrics into
counter rates and histogram quantiles. A supervisor (trimtabd) that owns the
engine process, applies hot fields live, and relaunches with new flags for
cold fields, with weights warm on local disk so the relaunch skips the image
pull and weight download that make a redeploy slow. GPU-resident reinit is
not claimed, neither engine exposes an in-process path for it yet.

Test status. 32 tests pass on CPU. Adapters, manifest, store, reconciler,
canary over a three-replica mock fleet, metrics parsing and quantiles,
supervisor against a real subprocess including cold relaunch and crash
recovery, and the CLI end to end including canary and supervise. The bench script and the on-pod runner both run in a
dry mode against the mock engine with zero dropped requests at 32 concurrent
load threads. GPU numbers, measured. Six cells, two engines on three GPUs (H100, RTX PRO
6000 Blackwell, B200), 20 of 20 swaps ok in every cell, zero dropped requests
under load in every cell, control call p50 12 to 20 ms. The table with every
column explained is in docs/results.md and the per-swap JSON is in
bench/results. docs/environment-notes.md records the environment problems the
vLLM lane hit and how the scripts handle them.

## Benchmarks

The short version. We started a real SGLang or vLLM server on one GPU, kept
it busy with 32 concurrent generation requests, and changed its concurrency
limit twenty times while it ran. The change was acknowledged in about 15 ms,
was in force in under a tenth of a second on SGLang, and not one request
failed. Doing the same thing today means a restart, which on these machines
takes one to seven minutes.

What the bench does. One GPU, Qwen/Qwen3.8-27B-FP8. 32 threads keep about 32
generation requests in flight the whole time. The bench sets the concurrency
limit to 8, then back to its boot value, twenty times, two seconds apart. For
each change it records the time until the server acknowledges the command
(API) and the time until the scheduler reports the new limit as in force
(effect). Generation requests that fail during the run are counted as dropped.

Columns. Boot cap is the normal limit the engine chose at startup, the value
we flip against. p50 and p95 are medians and 95th percentiles over the twenty
flips. Requests under load is how many generations the 32 threads completed
during the run.

| run | GPU | engine | boot cap | swaps ok | API p50 ms | API p95 ms | effect p50 ms | effect p95 ms | requests under load | dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| b200-sglang | NVIDIA B200 | sglang 0.5.18 | 80 | 20/20 | 16.64 | 47.96 | 78.39 | 123.82 | 806 | 0 |
| b200-vllm | NVIDIA B200 | vllm 0.28.0 | 256 | 20/20 | 11.54 | 17.61 | 649.49 | 1529.43 | 947 | 0 |
| h100-sglang | NVIDIA H100 80GB HBM3 | sglang 0.5.18 | 24 | 20/20 | 15.95 | 49.99 | 87.28 | 126.38 | 489 | 0 |
| h100-vllm | NVIDIA H100 80GB HBM3 | vllm 0.28.0 | 256 | 20/20 | 16.97 | 80.15 | 346.89 | 1296.47 | 531 | 0 |
| rtx6000-sglang | NVIDIA RTX PRO 6000 Blackwell Server Edition | sglang 0.5.18 | 33 | 20/20 | 19.36 | 31.97 | 78.72 | 99.72 | 429 | 0 |
| rtx6000-vllm | NVIDIA RTX PRO 6000 Blackwell Server Edition | vllm 0.28.0 | 256 | 20/20 | 20.21 | 40.47 | 1304.39 | 2663.86 | 444 | 0 |

What a restart costs on the same machines, for comparison. Cold disk is the
first boot after downloading weights, warm cache is the second boot of the
same weights. Warm cache is what trimtab's supervisor pays when a knob really
does need a relaunch.

| run | boot cold disk s | boot warm cache s |
|---|---|---|
| b200-sglang | 428 | 281 |
| b200-vllm | 166 | 86 |
| h100-sglang | 147 | 146 |
| h100-vllm | 321 | 75 |
| rtx6000-sglang | 141 | 56 |
| rtx6000-vllm | 60 | 55 |

Why vLLM's effect number is bigger. vLLM checks on every step that no more
requests are running than the limit allows, so a lower limit cannot fully
apply until enough in-flight requests finish. trimtab stops new admissions
immediately and tightens the enforced limit as requests complete, and the
bench measures the enforced value, so vLLM's number includes that drain.
Raising the limit is immediate on both engines. SGLang applies a lower limit
immediately and lets in-flight requests finish above it.

Per-swap rows and the HTTP status of every control call are in bench/results.
Reproduce a cell with one command.

```
bench/launch_runpod.sh h100 sglang
bench/launch_runpod.sh rtx6000 vllm
modal run bench/launch_modal_b200.py
modal run bench/launch_modal_b200_vllm.py
```

docs/environment-notes.md lists the environment problems these runs hit and
how the scripts handle each.

### Cold knobs

These are baked into GPU memory or graph capture at boot, so changing one
means a relaunch. The supervisor does that relaunch with weights already on
local disk, and maps each knob to its launch flag through a table in
trimtab/supervisor.py. A cold knob not in the table is refused, not guessed.

| engine | cold knobs the supervisor handles |
|---|---|
| sglang | tp_size, ep_size, quantization, kv_cache_dtype, mem_fraction_static, max_total_num_tokens, cuda_graph_max_bs, attention_backend, page_size, speculative_algorithm, speculative_draft_model_path |
| vllm | tensor_parallel_size, quantization, kv_cache_dtype, gpu_memory_utilization, max_model_len, cuda_graph_sizes, speculative_config |

Likely next hot knobs, none done. Schedule policy, speculative draft depth,
per-request deadlines, watermark thresholds, log level. Each needs a few
lines of patch and a verified per-step read.

## Quickstart

Apply the patch inside the engine container, restart the server once. Every
change after that needs no restart.

```
python3 engine/sglang/apply_patch.py      # or engine/vllm/apply_patch.py
```

Change a knob on the live server, directly

```
python3 -m trimtab.cli set --engine sglang --url http://localhost:30000 max_running_requests=64
python3 -m trimtab.cli set --engine vllm   --url http://localhost:8000  max_num_seqs=64
```

Or through the versioned path, with a daemon converging the replica

```
python3 -m trimtab.cli propose  --db t.db --group prod --reason "lower cap" max_running_requests=64
python3 -m trimtab.cli promote  --db t.db --group prod 1
python3 -m trimtab.cli daemon   --db t.db --group prod --engine sglang --url http://localhost:30000 --replica r0
python3 -m trimtab.cli rollback --db t.db --group prod 1
```

Canary a candidate on one replica, gates as data, promote on pass, revert on fail

```
python3 -m trimtab.cli canary start   --db t.db --group prod --engine sglang --replicas r0,r1,r2 \
    --metrics r0=http://a:30000/metrics r1=http://b:30000/metrics r2=http://c:30000/metrics \
    2 --scope r0 --window 300 --gates '[{"metric":"p99_ttft_ms","max_ratio":1.25},{"metric":"throughput","min_ratio":0.9}]'
python3 -m trimtab.cli canary observe --db t.db --group prod --engine sglang --replicas r0,r1,r2 --metrics ... 1
```

Run the engine under trimtabd, which relaunches it for cold fields and applies hot ones live

```
python3 -m trimtab.cli supervise --db t.db --group prod --engine sglang --model Qwen/Qwen3.8-27B-FP8 --port 30000 --replica r0
```

Measure it under load

```
python3 bench/hot_swap_bench.py --engine sglang --base http://localhost:30000 --iters 20
```

Try everything without a GPU

```
python3 -m trimtab.mock_engine --engine sglang --port 30000 &
python3 -m pytest tests/
DRY=1 ENGINE=sglang OUT=/tmp/r bash bench/pod_run.sh
```

GPU runs, one command each, results land in bench/results

```
bench/launch_runpod.sh h100 sglang
bench/launch_runpod.sh rtx6000 sglang
modal run bench/launch_modal_b200.py --engine vllm
```

## Layout

```
engine/<engine>/apply_patch.py   anchor-verified patcher for installed trees, the source of truth
engine/<engine>/patches/         derived patch files against pinned upstream commits
engine/<engine>/manifests/       knob classification per engine version
trimtab/                         adapters, manifest, store, reconciler, cli, mock engine
tests/                           runs on CPU against the mock engines
bench/                           hot swap bench, on-pod runner, runpod and modal launchers
docs/                            the spec
```

trimtab runs on any GPU the engine runs on. The patches are scheduler-level
Python with no kernels and no architecture-specific paths.

## Roadmap

The spec in docs/spec.md covers the full system. A supervisor that keeps
weights resident so cold knobs (pool sizes, KV dtype, parallelism) reinit in
seconds instead of redeploying in minutes. A control plane where every change
is versioned, canaried against preregistered gates, and rollbackable. An
upstream PR to SGLang so the hook lives in the engine itself.

## License and what is not in this repo

Everything in this repository is Apache 2.0. Engine patches, adapters,
manifests, the store, reconciler, canary, supervisor, CLI, mock engine, bench,
and the published results. Use it, fork it, ship it.

Some things we build on top of trimtab are not open source and are not in
this repository.

### Fleet management

Many clusters and replica groups under one desired-state view, with tenancy,
RBAC, SSO, and audit export.

### Hosted control plane

The store and reconciler run as a service with a Postgres backend. An agent
on each pod talks to it. You never run the plane yourself.

### Configuration recipes

A corpus of measured serving configurations per model and per GPU, with the
evidence attached, from our inference pricing work. Starting points that
already have the knobs where they should be.

### Canary policy library

Preregistered gate sets per traffic shape that the open canary consumes as
data.

### Support and engineering

Work on the open code under contract, including new engine versions and new
engines.

If you run inference at a scale where any of that matters, or you want the
hot path in an engine or engine version we do not cover yet, reach out.
Numinous Technology, hello@numinous.technology.
