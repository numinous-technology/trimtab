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

SGLang already ships a POST /set_internal_state endpoint wired end to end,
from HTTP through a fan-out communicator to every scheduler rank. Upstream
guards it with an allowlist of five niche keys. trimtab widens that allowlist
to the knobs operators actually sweep and applies validated values to the
live scheduler, where the loop reads them on the next step.

Knobs made hot by the v0 patch, each verified against the scheduler source

| knob | validation |
|---|---|
| max_running_requests | 1 up to the value allocated at boot |
| max_queued_requests | 0 or more |
| chunked_prefill_size | 1 or more |

## Quickstart

Apply the patch inside any SGLang container, then restart the server once.
Every change after that needs no restart.

```
python3 engine/sglang/apply_patch.py
```

Change a knob on the live server

```
curl -X POST http://localhost:30000/set_internal_state \
  -H "Content-Type: application/json" \
  -d '{"server_args": {"max_running_requests": 64}}'
```

Measure it under load

```
python3 bench/hot_swap_bench.py --base http://localhost:30000 --iters 20
```

The bench reports API latency, time to effect confirmed by reading the value
back from the scheduler, and background request failures across every swap.

## Layout

```
engine/sglang/patches/     patch series against pinned upstream commits
engine/sglang/apply_patch.py   anchor-verified patcher for installed trees
engine/sglang/manifests/   knob classification per engine version
bench/                     measurement harness and results
docs/                      the spec
```

## Roadmap

The spec in docs/spec.md covers the full system. A supervisor that keeps
weights resident so cold knobs (pool sizes, KV dtype, parallelism) reinit in
seconds instead of redeploying in minutes. A control plane where every change
is versioned, canaried against preregistered gates, and rollbackable. An
upstream PR to SGLang so the hook lives in the engine itself.

## License

Apache 2.0
