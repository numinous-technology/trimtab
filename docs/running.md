# Running trimtab against an unmerged engine

The engine patches are not upstream yet, so you apply them to the engine
installed in your container. The patcher edits the installed tree in place,
verifies every anchor, writes a `.trimtab-orig` backup next to each file, and
refuses rather than guess if the installed version differs from what the patch
expects. It is reversible.

The patches are written against pinned upstream commits, in the file names
under `engine/<engine>/patches`. As of this writing, SGLang `3315356` and
vLLM `c0adee9` (0.28.0). A nearby version usually still applies; a far one is
refused, and you regenerate with `engine/<engine>/regen_patch.sh` against your
checkout.

## SGLang

Install trimtab next to SGLang (same Python environment), then patch.

```
pip install -e /path/to/trimtab
python3 engine/sglang/apply_patch.py --check     # reports, changes nothing
python3 engine/sglang/apply_patch.py             # applies, writes backups
```

Launch the server. Add `--enable-memory-saver` if you want warm reinit (KV
pool resize with weights resident). Without it, hot knobs still work and warm
reinit refuses cleanly.

```
python3 -m sglang.launch_server --model-path <model> --host 0.0.0.0 --port 30000 \
    --enable-memory-saver
```

Now every knob change is live.

```
python3 -m trimtab.cli set --engine sglang --url http://localhost:30000 max_running_requests=64
python3 -m trimtab.cli set --engine sglang --url http://localhost:30000 schedule_policy=lpm
python3 -m trimtab.cli get --engine sglang --url http://localhost:30000
```

To revert the patch, restore the backups:

```
find $(python3 -c "import sglang,os;print(os.path.dirname(sglang.__file__))") \
    -name '*.trimtab-orig' -exec sh -c 'mv "$1" "${1%.trimtab-orig}"' _ {} \;
```

## vLLM

Two launch requirements, both the same gates vLLM puts on its own dev routes.
`VLLM_SERVER_DEV_MODE=1` enables the `/trimtab/*` routes. `--enable-sleep-mode`
lets warm reinit free the KV pool. Without sleep mode, hot knobs still work and
warm reinit refuses cleanly.

```
pip install -e /path/to/trimtab
python3 engine/vllm/apply_patch.py --check
python3 engine/vllm/apply_patch.py

VLLM_SERVER_DEV_MODE=1 vllm serve <model> --served-model-name default \
    --host 0.0.0.0 --port 8000 --enable-sleep-mode
```

```
python3 -m trimtab.cli set --engine vllm --url http://localhost:8000 max_num_seqs=64
python3 -m trimtab.cli get --engine vllm --url http://localhost:8000
```

Revert the same way, against the vllm package directory.

## Hardware notes

These come from the runs in docs/environment-notes.md and matter regardless of
trimtab.

The vLLM PyPI wheels need a host driver of 575 or newer (CUDA 12.9+). A 570
driver fails at engine-core init with "driver too old".

On RTX PRO 6000 (SM120), the vLLM wheel's FlashInfer does not recognise the
arch. Add `--attention-backend TRITON_ATTN` and set
`VLLM_USE_FLASHINFER_SAMPLER=0`.

Hybrid attention plus Mamba models (Qwen3.8-27B and similar) size Mamba state
per sequence, so vLLM's default `--max-num-seqs 1024` can exceed the Mamba
cache blocks. Pass a smaller `--max-num-seqs`. SGLang has the same class of
issue as `--max-mamba-cache-size`.

## Verifying the warm path

Warm reinit is measured by `bench/warm_reinit_bench.py`, which resizes the KV
pool three times on a live server and serves a generation after each.

```
python3 bench/warm_reinit_bench.py --engine sglang --base http://localhost:30000
python3 bench/warm_reinit_bench.py --engine vllm   --base http://localhost:8000
```

## Without a GPU

The mock engine speaks both control surfaces, so the whole control plane runs
on CPU.

```
python3 -m trimtab.mock_engine --engine sglang --port 30000 &
python3 -m trimtab.cli set --engine sglang --url http://localhost:30000 max_running_requests=32
python3 -m pytest tests/
```
