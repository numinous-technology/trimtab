# trimtab

trimtab makes inference engine configuration changeable while the engine runs.
A trim tab is the small surface on a ship's rudder that steers the rudder that
steers the ship. A patch of under a hundred lines steers engines that take
minutes to boot. The name is always lowercase.

Home is github.com/numinous-technology/trimtab. This document is the working
spec. Written 2026-08-31, revised 2026-09-01 after reading SGLang master.

## The problem

Every production inference engine (SGLang, vLLM, TRT-LLM, Dynamo) freezes its
operational parameters at startup. Changing a completion deadline, queue cap,
speculative depth, or batch limit means killing the process and redeploying.
Because the engine reloads model weights on boot, each change costs minutes.
Our measured SGLang cold boots run 90 to 252 seconds before counting image
pull and weight staging, and big MoE models are worse.

We lost entire working sessions this month to reboot-per-flag sweeps (mamba
cache size, speculative algorithm, chunked prefill, max running requests).
Operators complain about the same tax in public. The obvious wrong fix, moving
flags into a database without discipline, trades redeploys for unversioned
production roulette. The right fix is a control plane with versioning, canary,
and rollback sitting above deliberately dumb engine replicas.

## What trimtab does

Hot knobs change on a live engine in under 50 ms with no restart and no weight
reload. Cold knobs go through an in-place reinit with weights kept resident,
taking seconds instead of minutes. Every change is versioned, canaried on a
subset of replicas against preregistered pass and fail gates, and reversible
to any prior version.

## Hot and cold knobs

Every engine parameter is classified once, in a manifest, into exactly one
class. This classification is the heart of the system.

Hot knobs are pure scheduler-loop state. Nothing on the GPU is allocated when
they change, so they can change between two scheduling steps. The list
includes completion deadlines and per-request timeouts, admission queue depth,
max running requests as a logical cap, schedule policy, watermark thresholds,
chunked prefill size, speculative draft depth when the pipeline was sized to a
maximum at boot, and log verbosity.

Cold knobs are baked into GPU memory layout or graph capture at boot. The list
includes TP, PP, and EP degree, quantization, KV cache dtype, KV pool and
state pool sizes, the captured CUDA graph batch sizes, attention backend, and
the draft model itself. Cold changes need a reinit, but they never need a
weight reload. That separation is what makes them fast.

## What SGLang already ships, and what is missing

Reading SGLang master (commit 3315356, 2026-09-01) changed our plan. The
transport we expected to build already exists end to end.

SGLang has a POST /set_internal_state HTTP route, a declarative fan-out
communicator that carries the request from the tokenizer manager to every
scheduler rank, and a Scheduler.set_internal_state handler. The handler guards
an allowlist of exactly five niche keys (pp max micro batch size, two
speculative accept thresholds, two dspark controls) and routes accepted values
into a config override bag.

Two things keep this from being useful today. The allowlist omits every knob
an operator actually sweeps. And the useful knobs are snapshotted into
scheduler instance attributes at init, so a config-bag override never reaches
the running loop anyway. We verified the loop reads self.max_queued_requests
at admission (scheduler.py line 3077) and self.max_running_requests when
building each batch (lines 2274, 2302, 3626), which means mutating those
attributes on the live instance takes effect on the next scheduling step.

So the v0 engine patch is one function in one file. Widen the allowlist,
validate each value against a physical ceiling recorded at boot, and apply it
to the live scheduler attribute. Roughly 60 lines. The upstream PR writes
itself, since it extends an endpoint SGLang already documents with a curl
example in its own source.

## Architecture

Three tiers. The design rule is a thin in-engine patch and a fat external
plane.

```
CONTROL PLANE  (external service, engine agnostic)
  config store in Postgres, append-only, the table is the version history
  admin API over REST and gRPC
  canary orchestrator with auto-rollback
  metrics ingestion feeding promotion gates
  knob manifest per engine version
        |
        |  desired config (versioned) and control RPCs
        v
PER-POD SUPERVISOR  (trimtabd, wraps the engine, no fork)
  holds weights resident on GPU or pinned host memory
  forwards hot changes to the engine control endpoint
  runs cold changes as fast in-place reinits
  health, readiness, local last-known-good rollback cache
        |
        |  existing HTTP and ZMQ control surface
        v
ENGINE  (SGLang, small patch)
  set_internal_state extended to hot scheduler knobs
  values validated against physical ceilings recorded at boot
  boot-time over-provisioning so logical caps sit under fixed ceilings
```

## The engine patch

Three parts, applied to SGLang.

First, the widened set_internal_state. The handler accepts the hot knob set,
validates each value (max running requests may not exceed the boot value,
which is the physical ceiling backed by the allocated KV pool and captured
graphs), applies it to the live scheduler attribute, and reports what was
applied and rejected. The scheduler loop already reads these attributes every
step, so no other engine code changes.

Second, boot-time over-provisioning. At launch, allocate the physical ceiling
rather than the requested value. Capture CUDA graphs for the superset of batch
sizes we might ever cap to, size the KV and state pools to the maximum we
would sweep, and size the speculative pipeline to maximum depth. Hot knobs
then move freely under a fixed ceiling. Exposed as ceiling flags that default
to the requested values, so behavior without trimtab is unchanged.

Third, multi-rank consistency. TP greater than one means several scheduler
processes. The fan-out communicator already delivers the update to every
rank. Updates carry a version stamp so ranks converge on the same config even
if messages interleave with steps.

Patch budget is under 400 lines total across all three parts, and v0 (the
first part alone) is about 60. Small enough to rebase on upstream releases in
under a day, and shaped for an upstream PR.

## The supervisor

trimtabd is a small process that owns the engine lifecycle on each pod. Two
processes per pod, supervisor and engine.

For a hot diff it forwards the update to the engine control endpoint, confirms
the applied version, and reports back. Target under 50 ms end to end.

For a cold diff it drains in-flight requests (or hard-cuts when policy says
so), reinits the engine against weights that never left memory, replays the
readiness probe, and resumes traffic. Target single-digit seconds.

Weight residency is what makes the cold path fast. Weight load is the villain
in every redeploy. trimtabd keeps weights GPU-resident across a reinit when
VRAM allows, since a new memory layout can reuse the same weight tensors while
KV, state, and graph allocations are torn down and rebuilt. When VRAM must be
freed, weights fall back to pinned host memory and reload over PCIe in
seconds rather than from network or disk in minutes. A local content-addressed
weight cache is the backstop. The graph superset is captured once at boot so a
reinit does not pay full recapture.

Before we publish any cold-path number, we measure all three residency paths
(GPU-resident, pinned-host, and the quant-change worst case that invalidates
resident weights) and label each. We do not quote the best path as the
general case.

trimtabd keeps the last-known-good config locally so it can revert even when
the control plane is unreachable.

## The control plane service

The config store is Postgres, append-only. Every write is a new immutable
config version row scoped by model and replica group. The table is the version
history, which gives audit and rollback for free. A desired-state row points
each replica group at its target version, and supervisors reconcile toward it.
Reconciliation rather than push, because it survives partitions and maps
directly onto a Kubernetes operator later.

The knob manifest classifies every parameter of every supported engine version
as hot or cold, with validators for type, range, and cross-field constraints
such as a cap that may not exceed a ceiling. The manifest ships with the
engine adapter and is versioned with it.

The admin API covers set-config, get-config, list-versions, diff, rollback to
a named version, and canary start, promote, and abort. Every mutating call
records actor, reason, and timestamp.

Metrics ingestion scrapes the engine's server info endpoint and Prometheus
(throughput, TTFT p50 and p99, TPOT, queue depth, error rate, speculative
accept rate) and feeds the promotion gates.

## Canary and rollback

Live config without canary is production roulette, so promotion is gated.

A proposed config version applies first to a subset of the replica group,
instantly for hot and via fast reinit for cold. It is observed for a
configured window against the baseline on throughput delta, tail latency,
error rate, and accept rate, with preregistered pass and fail thresholds. A
config is a claim and the canary is its measurement, the same evidence
discipline we use everywhere else. On pass it promotes to the rest of the
group. On fail the canary replicas revert to last known good automatically.
Any promoted version can be rolled back by pointing desired state at a prior
version. Gate definitions live in the config store as data, so users express
policy without forking.

A shadow-traffic mode mirrors a fraction of real requests to a canary replica
and discards the responses, for zero-risk evaluation of risky cold changes.

## Data model sketch

```
config_version(id, model, replica_group, parent_id, klass, fields_jsonb,
               created_by, reason, created_at)          append-only
desired_state(replica_group, target_version_id, updated_at)
replica_status(replica_id, replica_group, applied_version_id, state,
               last_heartbeat)
canary(id, replica_group, candidate_version_id, baseline_version_id,
       scope, gates_jsonb, status, started_at, decided_at, decision)
knob_manifest(engine, engine_version, field, klass, validator_jsonb,
              ceiling_flag)
```

## Measurement plan

Preregister and measure on one real model. Start with Qwen3.8-27B on a single
H100, then a datacenter MoE.

Hot-swap latency, from API call to the change taking effect in the scheduler
loop, target under 50 ms. Cold-reinit latency on all three labeled residency
paths. A redeploy baseline on the same model for contrast, the minutes number.
Correctness, meaning zero dropped or corrupted requests across a hot swap and
across a drained cold reinit, checked by checksumming a request stream through
the change. And iteration throughput, configs evaluated per hour with trimtab
against the redeploy loop.

Every number is labeled measured, with engine commit, model revision,
hardware, and flags, per our evidence rules.

## Milestones

M0, taxonomy. Classify all SGLang knobs into the manifest with validators.
Done by reading scheduler source, no GPU needed. Partially complete, see the
set_internal_state findings above.

M1, hot path on a live engine. The widened set_internal_state patch plus a
bench proving a live max running requests and queue cap change under load,
with the redeploy baseline measured on the same pod. This alone is a
shippable, publishable result.

M2, trimtabd. Built as a supervisor that owns the engine process, applies hot
fields live, and relaunches with new flags for cold fields, with weights warm
on local disk. That is the pinned-host path. It skips the image pull and
weight download that make a redeploy slow, and its cost is one engine boot
against a warm page cache, which pod_run.sh measures as boot_patched_s next
to boot_stock_s (cold disk after download). GPU-resident reinit is not built.
Neither engine exposes an in-process pool rebuild, so claiming it would be a
lie. It stays on the roadmap as an upstream conversation.

M3, store with versioning and rollback. Built on SQLite with the schema the
Postgres deployment will share. The admin surface is the CLI. A network API
comes when a second machine needs to talk to the store.

M4, canary orchestrator. Built. Per-replica overrides put a candidate on a
subset, a Prometheus scraper turns engine metrics into rates and quantiles,
gates are data, pass promotes to the group, fail reverts, and every gate
produces a finding so a reviewer sees what was checked.

M5, vLLM. Done alongside M1 rather than after, since reading vLLM main showed
the same shape as SGLang (caps snapshotted at init, read per step, an existing
utility RPC dispatched by method name). Patch, manifest, adapter, and tests
exist. GPU verification pending with the rest.

## Open source plan

trimtab is open source under Apache 2.0. Adoption is the moat for a control
plane. The things worth money later (fleet features, hosted plane, our
measured config recipes) are not in the repo either way.

The upstream strategy is aggressive. The engine hook goes to SGLang as an RFC
and PR the same week the repo goes public, since it extends their existing
set_internal_state endpoint. Once merged, our rebase cost drops to zero and
every SGLang user has the socket trimtab speaks. We stay the reference
implementation of our own protocol.

What is open. Engine adapters and knob manifests, trimtabd, the single-cluster
control plane with store, versioning, rollback and CLI, and the bench harness
with published results. What stays ours. Multi-cluster and multi-tenant fleet
management, hosted control plane, and the recipe corpus of measured configs
from our pricing work.

The stack. Python throughout for v1. Both engine patches must be Python, and
one language means one test suite and a mock engine that exercises the real
adapter code. A Go static binary for the daemon was the original plan and is
deferred until operators ask for a venv-free install. Postgres for the store
in a deployment, SQLite with the same schema for dev and single node, so the
quickstart needs zero infrastructure. CUE for manifests if contributors tolerate it, YAML with JSON
schema if not. Prometheus for metrics. Ship both a docker-compose quickstart
and a Kubernetes operator, because half the audience lives on k8s and the
other half runs bare pods on RunPod and Modal and bounces off anything
k8s-only.

Repo layout.

```
trimtab/
  engine/
    sglang/
      patches/        patch series against pinned upstream SHAs
      manifests/      knob classification per engine version
    adapter.go        EngineAdapter interface (Apply hot, Reinit cold, Probe)
  trimtabd/           supervisor daemon
  controlplane/       API, store, canary orchestrator
  cli/                trimtab CLI, alias tt
  proto/              shared types, single source of truth
  bench/              measurement harness and published results
  deploy/
    compose/          zero-infra quickstart
    operator/         k8s CRD and controller
  docs/
```

A mock engine that speaks the same control surface without a GPU ships early,
so the whole plane is developed and CI-tested on CPU and contributors without
GPUs can work on everything except the patch itself.

Sequencing for release. Ship M1 with a recorded demo, a live flag change in
single-digit milliseconds next to a four-minute redeploy of the same server.
Open the SGLang RFC the same week. Publish the bench table with engine
commit, model, hardware, and all three cold paths labeled. Then build M3 and
M4 in public.

## Infrastructure needs

Docker GPU pods suffice. Nothing needs bare metal. The patch is scheduler
level Python, no kernels and no architecture-specific code, so trimtab runs on
any GPU SGLang supports. Hopper, Ada, Ampere, Blackwell, all fine. An H100
covers the demo, and Blackwell is only relevant when a chosen model needs it.

Most of the build is CPU. The control plane, trimtabd, CLI, manifests, and
mock engine develop and test on the box we already have, with Postgres we
already run. GPU spend is one small persistent RunPod pod during M1 and M2
iteration, a brief two-GPU pod for TP-rank consistency, and a few hours of
8-GPU time at the end for the public benchmark table on a datacenter model. No
S3. Weights stage from HF Hub to pod volumes as we already do.

## Risks

Upstream drift. SGLang moves fast. Mitigated by the tiny patch, the manifest
versioned per engine release, and the upstream PR strategy. If upstream ships
native live config, we adopt it and keep the plane.

Over-provisioning tax. A larger graph superset and maximum pools raise boot
time and floor VRAM. Measure the tax, pick a sane ceiling set, not everything.

Cold reinit semantics. Drain against hard-cut per knob. Some cold changes
cannot keep resident weights (quantization) and are honestly slower. Label
them.

Cross-field validity. Some hot values are only valid together. Manifest
validators encode the constraints.

Multi-rank consistency. Version-stamped updates applied on step boundaries,
verified in the TP-greater-than-one test before M1 is called done.

## Success criteria for v1

1. A hot knob changes on a live production-shaped SGLang server in under
   50 ms, measured, with zero dropped requests.
2. A cold knob changes via resident-weight reinit in single-digit seconds,
   measured against a minutes-long redeploy baseline on the same pod.
3. Every change is versioned, canaried against preregistered gates, and
   rollbackable to any prior version.
4. The engine patch rebases onto a newer SGLang release in under a day.
5. One published benchmark table with all three cold paths labeled.
