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

Two engines, one control surface. Both SGLang and vLLM snapshot their
scheduler caps into instance attributes at boot and read them on every step.
The patches make those attributes writable through each engine's existing
control RPC, validated against ceilings recorded at boot.

SGLang ships a POST /set_internal_state route wired end to end, from HTTP
through a fan-out communicator to every scheduler rank, guarded by an
allowlist of five niche keys. The patch widens the allowlist and applies the
values to the live scheduler. vLLM dispatches utility RPCs to EngineCore by
method name, so the patch adds one method and one dev router (the server runs
with VLLM_SERVER_DEV_MODE=1, the same gate vLLM puts on its own dev routes).

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
load threads. GPU numbers are not yet measured and nothing here claims them.

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

## License

Apache 2.0
