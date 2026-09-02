# trimtab

Change SGLang and vLLM scheduler settings while the server runs.

A trim tab is the small surface on a ship's rudder that steers the rudder
that steers the ship. trimtab is a small patch and a control plane that steer
inference engines without restarting them.

## The problem

Inference engines read their flags once at startup. To change a concurrency
cap, a queue limit, or a prefill chunk size you kill the process and reload
the weights. On the machines in this repo that costs one to seven minutes per
change. Tuning becomes an offline chore and every experiment pays a reboot.

Most of those settings are plain scheduler state. Nothing on the GPU is
allocated when they change. They are startup flags because nobody wired a
setter.

## How it works

When SGLang or vLLM starts, it copies a few flags into variables on the
scheduler object. From then on the flag is dead. The scheduler reads the
variable many times a second to decide what to run next. trimtab changes that
variable on the running server.

Each engine already has a way to send a command into the scheduler. SGLang has
a `POST /set_internal_state` route that fans out to every scheduler rank,
guarded by an allowlist of five niche keys. vLLM dispatches utility RPCs to
`EngineCore` by method name. The patch adds a handler on each side that takes
the new value, checks it against the ceiling the engine sized memory for at
boot, and writes it. The scheduler picks it up on its next step. The SGLang
patch is about 70 lines. The vLLM patch is about 90 and needs the server to
run with `VLLM_SERVER_DEV_MODE=1`, the same gate vLLM puts on its own dev
routes.

The patches are scheduler-level Python with no kernels and no
architecture-specific code, so trimtab runs on any GPU the engine runs on.

## What you can change

trimtab supports the knobs in the manifests and nothing else. Every other flag
is set at launch as before. Knobs come in three classes. Hot changes live.
Warm rebuilds the KV pool in place with weights resident. Cold relaunches.

### Hot knobs, changed live

A knob is hot only after the line where the scheduler reads it on every step
has been checked in the engine source. That line is the last column. Every
knob below was set, read back, and restored on a live engine on real hardware
(the knob sweep in the results).

| engine | knob | allowed values | per-step read |
|---|---|---|---|
| sglang 0.5.18 | max_running_requests | 1 to boot value | scheduler.py 2274, 2302, 3626 |
| sglang 0.5.18 | max_queued_requests | 0 or more | scheduler.py 3077 |
| sglang 0.5.18 | chunked_prefill_size | 1 or more | scheduler.py 3600 |
| sglang 0.5.18 | max_prefill_tokens | 1 to KV pool size | scheduler.py 3621 |
| sglang 0.5.18 | schedule_policy | fcfs, lpm, dfs-weight, lof, random, priority | schedule_policy.py 257, 294, 297 |
| sglang 0.5.18 | schedule_conservativeness | above 0, rebuilds the new-token-ratio watermarks | scheduler.py 3620, 3890 to 3948 |
| sglang 0.5.18 | log_level | DEBUG, INFO, WARNING, ERROR | every log call |
| vllm 0.28.0 | max_num_seqs | 1 to boot value | v1/core/sched/scheduler.py 792 |
| vllm 0.28.0 | max_num_batched_tokens | 1 to boot value | v1/core/sched/scheduler.py 440, 529 |
| vllm 0.28.0 | long_prefill_token_threshold | 0 or more | v1/core/sched/scheduler.py 441, 598, 992 |
| vllm 0.28.0 | log_level | DEBUG, INFO, WARNING, ERROR | every log call |

SGLang's five upstream keys (pp_max_micro_batch_size, two speculative accept
thresholds, two dspark controls) still work through the same route.

Knobs that look hot and are not, with the reason recorded in the manifest so
nobody re-investigates. Speculative depth on both engines, the draft CUDA
graphs are captured for a fixed depth at boot. SGLang's watchdog timeout, it
is passed by value into a thread at boot. vLLM's scheduling policy, the
waiting queue object is built for one policy at init and switching needs a
queue rebuild. Neither engine has a scheduler-level request deadline knob,
deadlines are per-request parameters.

### Warm knobs, rebuilt in place with weights resident

Both engines, measured. The engine drains, frees the KV pool and CUDA graphs,
rebuilds pools, backends and graphs at the new size, and rewires the
scheduler. The weights never move. SGLang needs `--enable-memory-saver`, vLLM
needs `--enable-sleep-mode`, both of which the supervisor and bench add.

On SGLang, `mem_fraction_static` and `max_total_num_tokens` (the
KV pool size), and raising the `max_running_requests` ceiling. The engine
drains, unmaps the old KV pool and CUDA graphs through its own memory saver,
rebuilds pools, attention backends and graphs at the new size, and rewires
every scheduler component to the new objects. The 29 GB of weights never move.
Needs the server launched with `--enable-memory-saver`, which the supervisor
and bench add for SGLang. The supervisor tries this path before a relaunch.

On vLLM, `gpu_memory_utilization` (the KV pool size), co-changing
`max_num_seqs` in the same rebuild.

Measured, three consecutive resizes each followed by a served generation.
SGLang on RTX PRO 6000: 31 to 32 s call to first token with prefill CUDA
graphs on (29 s of that is capturing them), 2.0 to 2.4 s with them off
(`--disable-prefill-cuda-graph`), freeing under 0.6 s. vLLM on H100: 8.3 to
10.1 s call to first token, freeing 35.7 GiB in ~0.2 s. A relaunch on the same
GPU is 56 to 86 s warm, 141 to 391 s cold.

### Cold knobs, changed by relaunch

These are existing engine flags baked into GPU memory or graph capture at
boot. trimtab does not make them hot. The supervisor knows which they are and,
when a desired version changes one, drains, stops the engine, relaunches with
the new flag, waits for health, and re-applies the hot knobs. A cold knob
missing from its table is refused, not guessed.

| engine | cold knobs the supervisor relaunches with |
|---|---|
| sglang | tp_size, ep_size, quantization, kv_cache_dtype, cuda_graph_max_bs, attention_backend, page_size, speculative_algorithm, speculative_draft_model_path |
| vllm | tensor_parallel_size, quantization, kv_cache_dtype, max_model_len, cuda_graph_sizes, speculative_config |

A redeploy is image pull, weight download, pod scheduling, and engine boot.
The supervisor's relaunch skips the first three and pays the boot against
weights already on local disk. Measured below, that boot took 55 to 281
seconds. These knobs change the weights' own layout (parallelism,
quantization) or the model itself, so the pool-only warm path cannot cover
them.

## Measured

We started a real SGLang or vLLM server on one GPU, kept it busy with 32
concurrent generation requests, and changed its concurrency cap twenty times
while it ran. The server acknowledged each change in about 15 ms. On SGLang
the new cap was in force in under a tenth of a second. Not one request failed.
A restart on the same machines takes one to seven minutes.

The bench. One GPU, Qwen/Qwen3.8-27B-FP8. 32 threads keep about 32
generation requests in flight. The bench sets the cap to 8, then back to its
boot value, twenty times, two seconds apart. For each change it records the
time until the server acknowledges the command (API) and the time until the
scheduler reports the new cap as in force (effect). Generation requests that
fail during the run count as dropped. p50 and p95 are over the twenty
changes.

| GPU | engine | boot cap | changes ok | API p50 ms | API p95 ms | effect p50 ms | effect p95 ms | requests completed | dropped |
|---|---|---|---|---|---|---|---|---|---|
| H100 80GB | sglang 0.5.18 | 24 | 20/20 | 16 | 50 | 87 | 126 | 489 | 0 |
| RTX PRO 6000 Blackwell | sglang 0.5.18 | 33 | 20/20 | 19 | 32 | 79 | 100 | 429 | 0 |
| B200 | sglang 0.5.18 | 80 | 20/20 | 17 | 48 | 78 | 124 | 806 | 0 |
| H100 80GB | vllm 0.28.0 | 256 | 20/20 | 17 | 80 | 347 | 1296 | 531 | 0 |
| RTX PRO 6000 Blackwell | vllm 0.28.0 | 256 | 20/20 | 20 | 40 | 1304 | 2664 | 444 | 0 |
| B200 | vllm 0.28.0 | 256 | 20/20 | 12 | 18 | 649 | 1529 | 947 | 0 |

vLLM's effect number is larger because vLLM asserts on every step that no
more requests are running than the cap allows. A lower cap cannot fully apply
until enough in-flight requests finish. trimtab stops new admissions at once
and tightens the enforced cap as requests complete. The bench measures the
enforced value, so the drain is in the number. Raising the cap is immediate on
both engines. SGLang applies a lower cap immediately and lets in-flight
requests finish above it.

What a restart costs on the same machines. Cold disk is the first boot after
downloading weights. Warm cache is the second boot of the same weights, which
is what the supervisor pays for a cold knob.

| GPU | engine | boot, cold disk | boot, warm cache |
|---|---|---|---|
| H100 80GB | sglang | 147 s | 146 s |
| RTX PRO 6000 Blackwell | sglang | 141 s | 56 s |
| B200 | sglang | 428 s | 281 s |
| H100 80GB | vllm | 321 s | 75 s |
| RTX PRO 6000 Blackwell | vllm | 60 s | 55 s |
| B200 | vllm | 166 s | 86 s |

Per-change rows and the HTTP status of every control call are in
`bench/results`. `docs/results.md` is generated from those files.
`docs/environment-notes.md` records the environment problems the runs hit
(image entrypoints, driver floors, SM120 attention, hybrid model caps) and how
the scripts handle each.

## Quickstart

Patch the engine inside its container and restart once. No change after that
needs a restart.

```
python3 engine/sglang/apply_patch.py
python3 engine/vllm/apply_patch.py
```

Change a knob on a live server.

```
python3 -m trimtab.cli set --engine sglang --url http://localhost:30000 max_running_requests=64
python3 -m trimtab.cli set --engine vllm   --url http://localhost:8000  max_num_seqs=64
```

Or version the change and let a daemon apply it. Every version is kept, and
rollback points the group at an older one. `--db` takes a SQLite path, a
`postgresql://` dsn, or the `http://` address of a trimtab.server.

```
python3 -m trimtab.cli propose  --db t.db --group prod --reason "lower cap" max_running_requests=64
python3 -m trimtab.cli promote  --db t.db --group prod 1
python3 -m trimtab.cli daemon   --db t.db --group prod --engine sglang --url http://localhost:30000 --replica r0
python3 -m trimtab.cli rollback --db t.db --group prod 1
```

Canary a candidate on one replica. Gates are data. Pass promotes to the group,
fail reverts the canary replica.

```
python3 -m trimtab.cli canary start --db t.db --group prod --engine sglang --replicas r0,r1,r2 \
    --metrics r0=http://a:30000/metrics r1=http://b:30000/metrics r2=http://c:30000/metrics \
    2 --scope r0 --window 300 \
    --gates '[{"metric":"p99_ttft_ms","max_ratio":1.25},{"metric":"throughput","min_ratio":0.9}]'
python3 -m trimtab.cli canary observe --db t.db --group prod --engine sglang --replicas r0,r1,r2 --metrics ... 1
```

Run the engine under the supervisor. Hot knobs apply live, cold knobs
relaunch.

```
python3 -m trimtab.cli supervise --db t.db --group prod --engine sglang \
    --model Qwen/Qwen3.8-27B-FP8 --port 30000 --replica r0
```

Run the store as a service so daemons on other machines can reach it.

```
python3 -m trimtab.server --db postgresql://user:pw@host/trimtab --port 7070 --token secret
python3 -m trimtab.cli daemon --db http://plane:7070 --token secret --group prod --engine sglang --url http://localhost:30000 --replica r0
```

On Kubernetes, an InferenceConfig object is the front door. The operator turns
it into a version (or a canary) in the store and writes the store's view back
into its status.

```
kubectl apply -f deploy/k8s/crd.yaml
kubectl apply -f deploy/k8s/controlplane.yaml
kubectl apply -f deploy/k8s/example-inferenceconfig.yaml
kubectl get inferenceconfigs
```

Measure a live server under load.

```
python3 bench/hot_swap_bench.py --engine sglang --base http://localhost:30000 --iters 20
```

Try all of it without a GPU. The mock engine speaks both control surfaces.

```
python3 -m trimtab.mock_engine --engine sglang --port 30000 &
python3 -m pytest tests/
DRY=1 ENGINE=sglang OUT=/tmp/r bash bench/pod_run.sh
```

Reproduce a measured cell. Results land in `bench/results`.

```
bench/launch_runpod.sh h100 sglang
bench/launch_runpod.sh rtx6000 vllm
modal run bench/launch_modal_b200.py
modal run bench/launch_modal_b200_vllm.py
```

## What is in the repo

```
engine/<engine>/apply_patch.py   patcher for installed trees, refuses when anchors do not match
engine/<engine>/patches/         patch files derived from the patcher against pinned upstream commits
engine/<engine>/manifests/       hot, cold, deferred classification per engine version
trimtab/adapters.py              one call shape for both engines
trimtab/manifest.py              loads manifests, validates values at the boundary
trimtab/store.py                 append-only SQLite versions, the table is the history
trimtab/reconciler.py            converges one replica on its desired version, no-op when there
trimtab/canary.py                candidate on a subset, gates as data, promote or revert
trimtab/metrics.py               Prometheus text to counter rates and histogram quantiles
trimtab/supervisor.py            owns the engine process, hot live, cold by relaunch
trimtab/server.py                the store over HTTP, and RemoteStore with the Store interface
trimtab/operator.py              Kubernetes operator, InferenceConfig to store, stdlib api client
trimtab/cli.py                   set, get, propose, promote, rollback, canary, daemon, supervise
trimtab/mock_engine.py           both control surfaces without a GPU
tests/                           41 tests on CPU, plus Postgres and kind-cluster tests that run when those exist
bench/                           bench, knob sweep, on-pod runner, RunPod and Modal launchers, results
deploy/k8s/                      CRD, control plane and operator manifests, example object
docs/upstream/                   ready-to-push patches for SGLang and vLLM with measured numbers in the messages
docs/                            spec, results, environment notes
```

Tests cover the adapters, manifest validation, the store on SQLite and on a
real Postgres, the network API and RemoteStore over the wire, the reconciler,
canary over a three-replica mock fleet, metrics parsing, the supervisor
against a real subprocess including cold relaunch and crash recovery, the
operator against a real kind cluster, and the CLI end to end. The mock was corrected twice from the GPU runs. SGLang returns
one bool per rank, and vLLM reports a target and an enforced cap separately.
The mock now does both.

## Cold knobs, changed by relaunch

These are existing engine flags baked into GPU memory or graph capture at
boot. trimtab does not make them hot. The supervisor knows which they are and,
when a desired version changes one, drains, stops the engine, relaunches with
the new flag, waits for health, and re-applies the hot knobs. A cold knob
missing from its table is refused, not guessed.

| engine | cold knobs the supervisor relaunches with |
|---|---|
| sglang | tp_size, ep_size, quantization, kv_cache_dtype, cuda_graph_max_bs, attention_backend, page_size, speculative_algorithm, speculative_draft_model_path |
| vllm | tensor_parallel_size, quantization, kv_cache_dtype, max_model_len, cuda_graph_sizes, speculative_config |

A redeploy is image pull, weight download, pod scheduling, and engine boot.
The supervisor's relaunch skips the first three and pays the boot against
weights already on local disk. Measured below, that boot took 55 to 281
seconds. These knobs change the weights' own layout (parallelism,
quantization) or the model itself, so the pool-only warm path cannot cover
them.

## Measured

We started a real SGLang or vLLM server on one GPU, kept it busy with 32
concurrent generation requests, and changed its concurrency cap twenty times
while it ran. The server acknowledged each change in about 15 ms. On SGLang
the new cap was in force in under a tenth of a second. Not one request failed.
A restart on the same machines takes one to seven minutes.

The bench. One GPU, Qwen/Qwen3.8-27B-FP8. 32 threads keep about 32
generation requests in flight. The bench sets the cap to 8, then back to its
boot value, twenty times, two seconds apart. For each change it records the
time until the server acknowledges the command (API) and the time until the
scheduler reports the new cap as in force (effect). Generation requests that
fail during the run count as dropped. p50 and p95 are over the twenty
changes.

| GPU | engine | boot cap | changes ok | API p50 ms | API p95 ms | effect p50 ms | effect p95 ms | requests completed | dropped |
|---|---|---|---|---|---|---|---|---|---|
| H100 80GB | sglang 0.5.18 | 24 | 20/20 | 16 | 50 | 87 | 126 | 489 | 0 |
| RTX PRO 6000 Blackwell | sglang 0.5.18 | 33 | 20/20 | 19 | 32 | 79 | 100 | 429 | 0 |
| B200 | sglang 0.5.18 | 80 | 20/20 | 17 | 48 | 78 | 124 | 806 | 0 |
| H100 80GB | vllm 0.28.0 | 256 | 20/20 | 17 | 80 | 347 | 1296 | 531 | 0 |
| RTX PRO 6000 Blackwell | vllm 0.28.0 | 256 | 20/20 | 20 | 40 | 1304 | 2664 | 444 | 0 |
| B200 | vllm 0.28.0 | 256 | 20/20 | 12 | 18 | 649 | 1529 | 947 | 0 |

vLLM's effect number is larger because vLLM asserts on every step that no
more requests are running than the cap allows. A lower cap cannot fully apply
until enough in-flight requests finish. trimtab stops new admissions at once
and tightens the enforced cap as requests complete. The bench measures the
enforced value, so the drain is in the number. Raising the cap is immediate on
both engines. SGLang applies a lower cap immediately and lets in-flight
requests finish above it.

What a restart costs on the same machines. Cold disk is the first boot after
downloading weights. Warm cache is the second boot of the same weights, which
is what the supervisor pays for a cold knob.

| GPU | engine | boot, cold disk | boot, warm cache |
|---|---|---|---|
| H100 80GB | sglang | 147 s | 146 s |
| RTX PRO 6000 Blackwell | sglang | 141 s | 56 s |
| B200 | sglang | 428 s | 281 s |
| H100 80GB | vllm | 321 s | 75 s |
| RTX PRO 6000 Blackwell | vllm | 60 s | 55 s |
| B200 | vllm | 166 s | 86 s |

Per-change rows and the HTTP status of every control call are in
`bench/results`. `docs/results.md` is generated from those files.
`docs/environment-notes.md` records the environment problems the runs hit
(image entrypoints, driver floors, SM120 attention, hybrid model caps) and how
the scripts handle each.

## Quickstart

Patch the engine inside its container and restart once. No change after that
needs a restart.

```
python3 engine/sglang/apply_patch.py
python3 engine/vllm/apply_patch.py
```

Change a knob on a live server.

```
python3 -m trimtab.cli set --engine sglang --url http://localhost:30000 max_running_requests=64
python3 -m trimtab.cli set --engine vllm   --url http://localhost:8000  max_num_seqs=64
```

Or version the change and let a daemon apply it. Every version is kept, and
rollback points the group at an older one. `--db` takes a SQLite path, a
`postgresql://` dsn, or the `http://` address of a trimtab.server.

```
python3 -m trimtab.cli propose  --db t.db --group prod --reason "lower cap" max_running_requests=64
python3 -m trimtab.cli promote  --db t.db --group prod 1
python3 -m trimtab.cli daemon   --db t.db --group prod --engine sglang --url http://localhost:30000 --replica r0
python3 -m trimtab.cli rollback --db t.db --group prod 1
```

Canary a candidate on one replica. Gates are data. Pass promotes to the group,
fail reverts the canary replica.

```
python3 -m trimtab.cli canary start --db t.db --group prod --engine sglang --replicas r0,r1,r2 \
    --metrics r0=http://a:30000/metrics r1=http://b:30000/metrics r2=http://c:30000/metrics \
    2 --scope r0 --window 300 \
    --gates '[{"metric":"p99_ttft_ms","max_ratio":1.25},{"metric":"throughput","min_ratio":0.9}]'
python3 -m trimtab.cli canary observe --db t.db --group prod --engine sglang --replicas r0,r1,r2 --metrics ... 1
```

Run the engine under the supervisor. Hot knobs apply live, cold knobs
relaunch.

```
python3 -m trimtab.cli supervise --db t.db --group prod --engine sglang \
    --model Qwen/Qwen3.8-27B-FP8 --port 30000 --replica r0
```

Run the store as a service so daemons on other machines can reach it.

```
python3 -m trimtab.server --db postgresql://user:pw@host/trimtab --port 7070 --token secret
python3 -m trimtab.cli daemon --db http://plane:7070 --token secret --group prod --engine sglang --url http://localhost:30000 --replica r0
```

On Kubernetes, an InferenceConfig object is the front door. The operator turns
it into a version (or a canary) in the store and writes the store's view back
into its status.

```
kubectl apply -f deploy/k8s/crd.yaml
kubectl apply -f deploy/k8s/controlplane.yaml
kubectl apply -f deploy/k8s/example-inferenceconfig.yaml
kubectl get inferenceconfigs
```

Measure a live server under load.

```
python3 bench/hot_swap_bench.py --engine sglang --base http://localhost:30000 --iters 20
```

Try all of it without a GPU. The mock engine speaks both control surfaces.

```
python3 -m trimtab.mock_engine --engine sglang --port 30000 &
python3 -m pytest tests/
DRY=1 ENGINE=sglang OUT=/tmp/r bash bench/pod_run.sh
```

Reproduce a measured cell. Results land in `bench/results`.

```
bench/launch_runpod.sh h100 sglang
bench/launch_runpod.sh rtx6000 vllm
modal run bench/launch_modal_b200.py
modal run bench/launch_modal_b200_vllm.py
```

## What is in the repo

```
engine/<engine>/apply_patch.py   patcher for installed trees, refuses when anchors do not match
engine/<engine>/patches/         patch files derived from the patcher against pinned upstream commits
engine/<engine>/manifests/       hot, cold, deferred classification per engine version
trimtab/adapters.py              one call shape for both engines
trimtab/manifest.py              loads manifests, validates values at the boundary
trimtab/store.py                 append-only SQLite versions, the table is the history
trimtab/reconciler.py            converges one replica on its desired version, no-op when there
trimtab/canary.py                candidate on a subset, gates as data, promote or revert
trimtab/metrics.py               Prometheus text to counter rates and histogram quantiles
trimtab/supervisor.py            owns the engine process, hot live, cold by relaunch
trimtab/server.py                the store over HTTP, and RemoteStore with the Store interface
trimtab/operator.py              Kubernetes operator, InferenceConfig to store, stdlib api client
trimtab/cli.py                   set, get, propose, promote, rollback, canary, daemon, supervise
trimtab/mock_engine.py           both control surfaces without a GPU
tests/                           41 tests on CPU, plus Postgres and kind-cluster tests that run when those exist
bench/                           bench, knob sweep, on-pod runner, RunPod and Modal launchers, results
deploy/k8s/                      CRD, control plane and operator manifests, example object
docs/upstream/                   ready-to-push patches for SGLang and vLLM with measured numbers in the messages
docs/                            spec, results, environment notes
```

Tests cover the adapters, manifest validation, the store on SQLite and on a
real Postgres, the network API and RemoteStore over the wire, the reconciler,
canary over a three-replica mock fleet, metrics parsing, the supervisor
against a real subprocess including cold relaunch and crash recovery, the
operator against a real kind cluster, and the CLI end to end. The mock was corrected twice from the GPU runs. SGLang returns
one bool per rank, and vLLM reports a target and an enforced cap separately.
The mock now does both.

## License

Everything in this repository is Apache 2.0. Engine patches, adapters,
manifests, store, reconciler, canary, supervisor, CLI, mock engine, bench, and
the measured results.

## Not in this repo

Some things we build on top of trimtab are closed.

### Fleet management

Many clusters and replica groups under one desired-state view, with tenancy,
RBAC, SSO, and audit export.

### Hosted control plane

The store and reconciler as a service with a Postgres backend. An agent on
each pod talks to it. You do not run the plane.

### Configuration recipes

Measured serving configurations per model and per GPU with the evidence
attached, from our inference pricing work.

### Canary policy library

Gate sets per traffic shape that the open canary consumes as data.

### Support and engineering

Work on the open code under contract, including new engine versions and new
engines.

If you run inference at a scale where any of that matters, or need the hot
path in an engine or version we do not cover, write to
hello@numinous.technology.
