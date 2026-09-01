#!/bin/bash
# On-pod runner. Same script for every GPU and both engines.
#
#   ENGINE=sglang|vllm  MODEL=Qwen/Qwen3.8-27B-FP8  bash pod_run.sh
#
# Phases, each timed and written to /root/trimtab_results/<engine>.json
#   1 download weights
#   2 stock boot (no patch), to health, then kill        redeploy baseline A
#   3 apply patch, boot again to health                  redeploy baseline B
#   4 hot_swap_bench under load
#   5 nvidia-smi name, engine version, commit recorded alongside
set -u
ENGINE=${ENGINE:?}; MODEL=${MODEL:-Qwen/Qwen3.8-27B-FP8}
OUT=${OUT:-/root/trimtab_results}; mkdir -p $OUT; R=$OUT/$ENGINE.json
PORT=$([ $ENGINE = vllm ] && echo 8000 || echo 30000)
HERE=$(cd "$(dirname "$0")/.." && pwd)
MD=/workspace/$(echo $MODEL | tr / _)
log(){ echo "[$(date +%H:%M:%S)] $*"; }

if [ "${DRY:-0}" != 1 ] && [ $ENGINE = vllm ] && ! python3 -c "import vllm" 2>/dev/null; then
  t0=$(date +%s); pip install -q vllm huggingface_hub > $OUT/pip_vllm.log 2>&1 || { log "vllm install FAILED"; tail -20 $OUT/pip_vllm.log; exit 1; }
  log "vllm installed in $(( $(date +%s) - t0 ))s, $(python3 -c 'import vllm;print(vllm.__version__)')"
fi
if [ "${DRY:-0}" = 1 ]; then echo 0 > $OUT/download_s; else
python3 - <<EOF
import huggingface_hub, time; t=time.time()
huggingface_hub.snapshot_download("$MODEL", local_dir="$MD")
open("$OUT/download_s","w").write(str(round(time.time()-t)))
EOF
fi
log "weights at $MD ($(cat $OUT/download_s)s)"

boot(){ # $1 label -> writes $OUT/boot_$1_s, returns 1 on failure
  pkill -9 -f "launch_[s]erver|vllm [s]erve|vllm.[e]ntrypoints|[V]LLM|[E]ngineCore|trimtab.[m]ock_engine" 2>/dev/null; sleep ${KILL_WAIT:-5}
  if [ "${DRY:-0}" != 1 ]; then  # wait until the previous engine's GPU memory is really gone
    for _ in $(seq 1 30); do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ "${used:-0}" -lt 2000 ] && break; sleep 2
    done
    log "gpu memory used before boot $1: ${used}MiB"
  fi
  local t0=$(date +%s)
  if [ "${DRY:-0}" = 1 ]; then
    python3 -m trimtab.mock_engine --engine $ENGINE --port $PORT > $OUT/server_$1.log 2>&1 &
  elif [ $ENGINE = sglang ]; then
    python3 -m sglang.launch_server --model-path $MD --host 0.0.0.0 --port $PORT \
      --mem-fraction-static 0.85 > $OUT/server_$1.log 2>&1 &
  else
    VLLM_SERVER_DEV_MODE=1 vllm serve $MD --served-model-name default --host 0.0.0.0 --port $PORT \
      --gpu-memory-utilization 0.85 --max-model-len 16384 > $OUT/server_$1.log 2>&1 &
  fi
  while true; do
    c=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)
    [ "$c" = "200" ] && break
    grep -qE "Scheduler hit an exception|CUDA out of memory|Error|Traceback" $OUT/server_$1.log && sleep 20 && \
      { c=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health); [ "$c" = "200" ] || { log "boot $1 FAILED"; grep -aE "Error|error|rror:|Traceback|OOM|memory" $OUT/server_$1.log | head -40; tail -20 $OUT/server_$1.log; return 1; }; break; }
    [ $(( $(date +%s) - t0 )) -gt 1800 ] && { log "boot $1 TIMEOUT"; return 1; }
    sleep 5
  done
  echo $(( $(date +%s) - t0 )) > $OUT/boot_$1_s
  log "boot $1 ready in $(cat $OUT/boot_$1_s)s"
}

if [ "${SKIP_BOOT:-0}" = 1 ]; then log "SKIP_BOOT, using the running patched server"; else
boot stock || exit 1
if [ "${DRY:-0}" = 1 ]; then echo "dry run, patch skipped" > $OUT/patch.log; else
python3 $HERE/engine/$ENGINE/apply_patch.py | tee $OUT/patch.log; fi
boot patched || exit 1
fi
python3 $HERE/bench/hot_swap_bench.py --engine $ENGINE --base http://127.0.0.1:$PORT --iters ${ITERS:-20} --settle-s ${SETTLE:-10} --gap-s ${GAP:-2} --out $OUT/hot_swap.json
BENCH_EXIT=$?

python3 - <<EOF
import json, subprocess
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
r = {
  "engine": "$ENGINE", "model": "$MODEL",
  "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
  "engine_version": sh("python3 -c 'import $ENGINE; print($ENGINE.__version__)' 2>/dev/null") or "dry",
  "download_s": int(open("$OUT/download_s").read()),
  "boot_stock_s": int(open("$OUT/boot_stock_s").read()),
  "boot_patched_s": int(open("$OUT/boot_patched_s").read()),
  "bench_exit": $BENCH_EXIT,
  "hot_swap": json.load(open("$OUT/hot_swap.json")),
}
json.dump(r, open("$R", "w"), indent=1)
print(json.dumps({k: v for k, v in r.items() if k != "hot_swap"}, indent=1))
print(json.dumps({k: v for k, v in r["hot_swap"].items() if k != "rows"}, indent=1))
EOF
[ "${KEEP_SERVER:-0}" = 1 ] || pkill -9 -f "launch_[s]erver|vllm [s]erve|vllm.[e]ntrypoints|[V]LLM|[E]ngineCore|trimtab.[m]ock_engine" 2>/dev/null
log "done, results in $R"
