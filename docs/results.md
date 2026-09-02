# Measured results

Model Qwen/Qwen3.8-27B-FP8 on one GPU. Each row is one run of bench/pod_run.sh.
Every number is `measured`. The JSON behind each row is in bench/results with
per-swap rows and the HTTP status of every control call.

Hot knobs proven is the knob sweep, which sets every hot knob in the
manifest on the live engine, reads it back, and restores it. Cells run before
the sweep existed say "cap only", meaning only the concurrency cap was
exercised there.

Warm reinit is the slowest of three in-place KV pool rebuilds on the live
server (weights stay on the GPU), from the control call to the first token
served afterwards. n/a where the engine has no warm path yet.

What the columns mean. The bench runs 32 concurrent generation threads, then
flips the running-request cap between 8 and its boot value twenty times.
API is the control call round trip including the engine confirming on every
rank. Effect is from the call until the value read back from the scheduler
equals the target, which includes the read-back polling itself, so the true
in-engine latency is below it. Dropped counts background requests that failed
during the run. Boot cold disk is the first boot after weight download, boot
warm cache is the second boot of the same weights, which is the cost of the
supervisor's cold-knob relaunch path.

| run | GPU | engine | boot cap | swaps ok | API p50 ms | API p95 ms | effect p50 ms | effect p95 ms | requests under load | dropped | hot knobs proven | warm reinit | boot cold disk s | boot warm cache s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b200-sglang | NVIDIA B200 | sglang 0.5.18 | 80 | 20/20 | 16.64 | 47.96 | 78.39 | 123.82 | 806 | 0 | cap only | n/a | 428 | 281 |
| b200-vllm | NVIDIA B200 | vllm 0.28.0 | 256 | 20/20 | 24.36 | 82.08 | 140.41 | 1114.59 | 661 | 0 | 4/4 | n/a | 569 | 213 |
| h100-sglang | NVIDIA H100 80GB HBM3 | sglang 0.5.18 | 24 | 20/20 | 15.95 | 49.99 | 87.28 | 126.38 | 489 | 0 | cap only | n/a | 147 | 146 |
| h100-vllm | NVIDIA H100 80GB HBM3 | vllm 0.28.0 | 256 | 20/20 | 16.97 | 80.15 | 346.89 | 1296.47 | 531 | 0 | cap only | n/a | 321 | 75 |
| rtx6000-sglang | NVIDIA RTX PRO 6000 Blackwell Server Edition | sglang 0.5.18 | 33 | 20/20 | 16.78 | 29.73 | 66.99 | 100.01 | 424 | 0 | 7/7 | 32.1 s | 56 | 56 |
| rtx6000-vllm | NVIDIA RTX PRO 6000 Blackwell Server Edition | vllm 0.28.0 | 256 | 20/20 | 20.21 | 40.47 | 1304.39 | 2663.86 | 444 | 0 | cap only | n/a | 60 | 55 |

Effect latency differs by engine for a reason. SGLang applies a lowered cap
immediately and lets in-flight requests finish above it. vLLM asserts
running <= cap on every step, so trimtab stops admissions at once and
tightens the enforced cap as requests finish. The bench measures the enforced
value, so vLLM's effect number includes that drain and is dominated by output
length, not by the control path (its API latency is the same ~20 ms). Raising
the cap is immediate on both engines.

GPU-resident reinit, measured separately on RTX PRO 6000 with SGLang 0.5.18,
in bench/results/rtx6000-sglang-warm*.json. Three consecutive KV pool resizes
(mem_fraction 0.85 to 0.75, then an explicit 111k-token pool with cap 16,
then back) with weights resident, each followed by a served generation.
With prefill CUDA graphs on, 31.4 to 32.0 s call to first token, 29 s of it
prefill graph capture. With prefill CUDA graphs off, 2.0 to 2.4 s. Freeing
the old pool took 0.08 to 0.57 s in every case.

Not claimed. Warm reinit on vLLM. Anything on models or GPUs not in this table.
