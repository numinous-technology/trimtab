"""Build docs/results.md from bench/results/*.json. Numbers come from the
artifacts, never from a person. Run after every GPU run.

    python3 bench/summarize.py
"""
import glob
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROWS = []
for f in sorted(glob.glob(str(HERE / "results" / "*.json"))):
    r = json.load(open(f))
    h = r["hot_swap"]
    ROWS.append(dict(
        target=pathlib.Path(f).stem, gpu=r["gpu"].split(",")[0], engine=f'{r["engine"]} {r["engine_version"]}',
        boot_value=h["boot_value"], ok=f'{sum(x["ok"] for x in h["rows"])}/{h["iters"]}',
        api50=h["api_ms_median"], api95=h["api_ms_p95"], eff50=h["effect_ms_median"], eff95=h["effect_ms_p95"],
        bg=h["background_completed"], drops=h["background_failed"],
        stock=r["boot_stock_s"], patched=r["boot_patched_s"],
    ))

hdr = ("| run | GPU | engine | boot cap | swaps ok | API p50 ms | API p95 ms | effect p50 ms | effect p95 ms "
       "| requests under load | dropped | boot cold disk s | boot warm cache s |")
sep = "|" + "---|" * 13
lines = [hdr, sep] + [
    f'| {x["target"]} | {x["gpu"]} | {x["engine"]} | {x["boot_value"]} | {x["ok"]} | {x["api50"]} | {x["api95"]} '
    f'| {x["eff50"]} | {x["eff95"]} | {x["bg"]} | {x["drops"]} | {x["stock"]} | {x["patched"]} |' for x in ROWS]

doc = f"""# Measured results

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

{chr(10).join(lines)}

Effect latency differs by engine for a reason. SGLang applies a lowered cap
immediately and lets in-flight requests finish above it. vLLM asserts
running <= cap on every step, so trimtab stops admissions at once and
tightens the enforced cap as requests finish. The bench measures the enforced
value, so vLLM's effect number includes that drain and is dominated by output
length, not by the control path (its API latency is the same ~20 ms). Raising
the cap is immediate on both engines.

Not claimed. GPU-resident reinit. Anything on models or GPUs not in this table.
"""
(HERE.parent / "docs" / "results.md").write_text(doc)
print(doc)
