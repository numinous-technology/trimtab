# Measured results

Model Qwen/Qwen3.8-27B-FP8 on one GPU. Each row is one run of bench/pod_run.sh.
Every number is `measured`. The JSON behind each row is in bench/results with
per-swap rows and the HTTP status of every control call.

What the columns mean. The bench runs 32 concurrent generation threads, then
flips the running-request cap between 8 and its boot value twenty times.
API is the control call round trip including the engine confirming on every
rank. Effect is from the call until the value read back from the scheduler
equals the target, which includes the read-back polling itself, so the true
in-engine latency is below it. Dropped counts background requests that failed
during the run. Boot cold disk is the first boot after weight download, boot
warm cache is the second boot of the same weights, which is the cost of the
supervisor's cold-knob relaunch path.

| run | GPU | engine | boot cap | swaps ok | API p50 ms | API p95 ms | effect p50 ms | effect p95 ms | requests under load | dropped | boot cold disk s | boot warm cache s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b200-sglang | NVIDIA B200 | sglang 0.5.18 | 80 | 20/20 | 16.64 | 47.96 | 78.39 | 123.82 | 806 | 0 | 428 | 281 |
| h100-sglang | NVIDIA H100 80GB HBM3 | sglang 0.5.18 | 24 | 20/20 | 15.95 | 49.99 | 87.28 | 126.38 | 489 | 0 | 147 | 146 |
| rtx6000-sglang | NVIDIA RTX PRO 6000 Blackwell Server Edition | sglang 0.5.18 | 33 | 20/20 | 19.36 | 31.97 | 78.72 | 99.72 | 429 | 0 | 141 | 56 |

Not claimed. GPU-resident reinit. Anything on models or GPUs not in this table.
