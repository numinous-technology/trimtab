"""Run the M1 bench on a Modal B200, both engines.

    modal run bench/launch_modal_b200.py --engine sglang
    modal run bench/launch_modal_b200.py --engine vllm

Ships the repo into the container, runs bench/pod_run.sh unchanged, and
writes bench/results/b200-<engine>.json locally. Weights are cached in a
Modal volume across runs.
"""
import json
import os
import pathlib
import subprocess

import modal

HERE = pathlib.Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3.8-27B-FP8"
WEIGHTS = modal.Volume.from_name("weights-cache", create_if_missing=True)
IGNORE = [".git", "bench/results", "__pycache__", "*.pyc"]

sglang_image = (modal.Image.from_registry("lmsysorg/sglang:latest", add_python=None)
                .run_commands("python3 -m pip install -q huggingface_hub")
                .add_local_dir(str(HERE), "/root/trimtab", copy=True, ignore=IGNORE))
vllm_image = (modal.Image.from_registry("vllm/vllm-openai:latest", add_python=None).entrypoint([])
              .run_commands("python3 -m pip install -q huggingface_hub")
              .add_local_dir(str(HERE), "/root/trimtab", copy=True, ignore=IGNORE))

app = modal.App("trimtab-b200")


def run_bench(engine):
    env = {**os.environ, "ENGINE": engine, "MODEL": MODEL, "PYTHONPATH": "/root/trimtab"}
    r = subprocess.run(["bash", "bench/pod_run.sh"], cwd="/root/trimtab", capture_output=True, text=True, env=env)
    WEIGHTS.commit()
    out = {"log_tail": r.stdout[-6000:] + r.stderr[-3000:], "exit": r.returncode}
    try:
        out["result"] = json.load(open(f"/root/trimtab_results/{engine}.json"))
    except Exception as e:
        out["error"] = str(e)
    return out


@app.function(gpu="B200", image=sglang_image, timeout=60 * 90, volumes={"/workspace": WEIGHTS}, cpu=16.0, memory=131072)
def bench_sglang():
    return run_bench("sglang")


@app.function(gpu="B200", image=vllm_image, timeout=60 * 90, volumes={"/workspace": WEIGHTS}, cpu=16.0, memory=131072)
def bench_vllm():
    return run_bench("vllm")


@app.local_entrypoint()
def main(engine: str = "sglang"):
    out = {"sglang": bench_sglang, "vllm": bench_vllm}[engine].remote()
    res = HERE / "bench" / "results"
    res.mkdir(exist_ok=True)
    (res / f"b200-{engine}.log").write_text(out["log_tail"])
    if "result" in out:
        json.dump(out["result"], open(res / f"b200-{engine}.json", "w"), indent=1)
        r = out["result"]
        print(json.dumps({k: v for k, v in r.items() if k != "hot_swap"}, indent=1))
        print(json.dumps({k: v for k, v in r["hot_swap"].items() if k != "rows"}, indent=1))
    else:
        print("no result", out.get("error"), out["log_tail"][-2500:])
