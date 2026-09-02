# Environment notes from the GPU runs

Things that cost hours and were not trimtab bugs. Each is now handled in
bench/pod_run.sh or bench/launch_runpod.sh, recorded here so the next person
recognises the symptom.

## vllm/vllm-openai has an ENTRYPOINT

The official image sets `ENTRYPOINT ["vllm", "serve"]`. Anything you pass as
the container command becomes arguments to `vllm serve`, so an sshd bootstrap
never runs and the container crash-loops with no runtime. Symptom on RunPod,
the pod shows RUNNING for ever with no port mappings. The RunPod vLLM lane uses
a plain PyTorch image and installs vLLM on the pod. On Modal, the image works
once `.entrypoint([])` is set.

## vLLM wheels need a host driver of 575 or newer

PyPI vLLM 0.28 pulls torch built for CUDA 13. On a host with driver 570
(CUDA 12.8) the engine core dies with `The NVIDIA driver on your system is too
old (found version 12080)`. RunPod secure cloud mixes hosts with 570 and 580
drivers. The launcher reads the driver after SSH comes up and exits with code
3 on a too-old host so the pod is released and the run retried elsewhere.

## FlashInfer in the vLLM wheel does not know SM120

On RTX PRO 6000 (compute capability 12.0) the engine dies with `FlashInfer
requires GPUs with sm75 or higher`, which is a misleading message from a
capability table that stops before 12.x. `--attention-backend TRITON_ATTN`
plus `VLLM_USE_FLASHINFER_SAMPLER=0` boots and serves. pod_run.sh applies both
when compute capability starts with 12.

## Hybrid models and vLLM's default max_num_seqs

Qwen3.8-27B mixes attention and Mamba layers. vLLM sizes Mamba state per
sequence, and the default `max_num_seqs=1024` exceeds the blocks it can fit
(`max_num_seqs (1024) exceeds available Mamba cache blocks (978)`). pod_run.sh
passes `--max-num-seqs 256`. The same class of issue exists in SGLang as
`--max-mamba-cache-size`, where a value tuned for a 96 GB card starves the KV
pool on an 80 GB card.

## vLLM asserts running <= max_num_running_reqs every step

Lowering the cap below current occupancy through a plain attribute write kills
the engine core with an AssertionError on the next step. The trimtab patch
therefore applies a lowering as "stop admitting now" and tightens the enforced
cap in `schedule()` as requests finish. `/trimtab/knobs` reports both the
target (`max_num_seqs`) and the enforced value (`max_num_seqs_effective`).
SGLang has no such assertion and applies a lowering immediately.

## Killing an engine does not free its GPU memory at once

vLLM's EngineCore runs as separate processes that outlive the API server.
Relaunching five seconds after `pkill` can OOM against memory the old core
still holds. pod_run.sh kills the core children by name and waits for
`nvidia-smi` memory to drop below 2 GB before booting again.

## pkill -f matches the shell that runs it

A pattern like `pkill -f "vllm serve"` matches the command line of the shell
that contains the pkill, and kills it. Every pattern in these scripts uses the
bracket idiom (`vllm [s]erve`) so it cannot match its own invocation.
