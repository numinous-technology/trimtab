"""Apply the trimtab hot-knob patch to an installed vLLM tree.

vLLM's EngineCore already dispatches utility RPCs by method name
(getattr(self, method_name)), so hot knobs need one new method on EngineCore
and one dev router that calls it. The scheduler reads
max_num_running_reqs and max_num_scheduled_tokens on every step, verified in
vllm/v1/core/sched/scheduler.py.

The router registers under the dev API, so the server must run with
VLLM_SERVER_DEV_MODE=1. That is the same gate vLLM puts on its own
reset_prefix_cache, sleep, and collective_rpc routes.

Usage
    python3 apply_patch.py            apply
    python3 apply_patch.py --check    report status, change nothing
"""
import ast
import importlib.util
import shutil
import sys

MARKER = "trimtab_set_knobs"

CORE_ANCHOR = "    def _reject_add_in_shutdown(self, request: Request) -> bool:\n"

_HERE = __file__.rsplit("/", 1)[0]
CORE_REINIT = open(f"{_HERE}/reinit_core_body.py").read()
WORKER_ANCHOR = "    def sleep(self, level: int = 1) -> None:\n"
WORKER_BODY = open(f"{_HERE}/reinit_worker_body.py").read() + WORKER_ANCHOR

CORE_BODY = CORE_REINIT + '''    def trimtab_set_knobs(self, knobs: dict) -> dict:
        """trimtab (github.com/numinous-technology/trimtab) hot scheduler knobs.

        Applies validated values to the live scheduler, which reads them on
        the next step. Ceilings are the values allocated at boot.
        """
        sched = self.scheduler
        if not hasattr(sched, "_trimtab_ceilings"):
            sched._trimtab_ceilings = {
                "max_num_seqs": sched.max_num_running_reqs,
                "max_num_batched_tokens": sched.max_num_scheduled_tokens,
            }
        applied, rejected = {}, {}
        for key, value in knobs.items():
            if key == "long_prefill_token_threshold":
                if not isinstance(value, (int, float)) or int(value) < 0:
                    rejected[key] = "must be >= 0 (0 disables the threshold)"
                else:
                    sched.scheduler_config.long_prefill_token_threshold = int(value)
                    applied[key] = int(value)
                continue
            if key == "log_level":
                level = str(value).upper()
                if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
                    rejected[key] = "must be DEBUG, INFO, WARNING or ERROR"
                else:
                    import logging as _logging

                    _logging.getLogger("vllm").setLevel(level)
                    sched._trimtab_log_level = level
                    applied[key] = level
                continue
            if key not in sched._trimtab_ceilings:
                rejected[key] = "unknown knob"
                continue
            ceiling = sched._trimtab_ceilings[key]
            if not isinstance(value, (int, float)) or not (1 <= int(value) <= ceiling):
                rejected[key] = f"valid range is 1..{ceiling} (the boot allocation)"
                continue
            if key == "max_num_seqs":
                # The scheduler asserts len(running) <= cap every step, so a cap
                # below current occupancy is applied as "stop admitting now" and
                # tightened to the target in schedule() as requests finish.
                target = int(value)
                sched._trimtab_pending_max_num_seqs = target
                sched.max_num_running_reqs = max(target, len(sched.running))
            else:
                sched.max_num_scheduled_tokens = int(value)
            applied[key] = int(value)
        return {"ok": not rejected, "applied": applied, "rejected": rejected}

    def trimtab_get_knobs(self) -> dict:
        sched = self.scheduler
        pending = getattr(sched, "_trimtab_pending_max_num_seqs", None)
        return {
            "max_num_seqs": pending if pending is not None else sched.max_num_running_reqs,
            "max_num_seqs_effective": sched.max_num_running_reqs,
            "max_num_batched_tokens": sched.max_num_scheduled_tokens,
            "long_prefill_token_threshold": sched.scheduler_config.long_prefill_token_threshold,
            "log_level": getattr(sched, "_trimtab_log_level", None),
            "running": len(sched.running),
            "ceilings": getattr(sched, "_trimtab_ceilings", {}),
            "last_reinit": getattr(self, "_trimtab_last_reinit", None),
            "num_gpu_blocks": self.vllm_config.cache_config.num_gpu_blocks,
        }

'''

ROUTER_FILE = '''# SPDX-License-Identifier: Apache-2.0
# trimtab (github.com/numinous-technology/trimtab) dev router.

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

router = APIRouter()


def _core(request: Request):
    return request.app.state.engine_client.engine_core


@router.post("/trimtab/set_knobs")
async def trimtab_set_knobs(raw_request: Request):
    knobs = await raw_request.json()
    result = await _core(raw_request).call_utility_async("trimtab_set_knobs", knobs)
    return JSONResponse(content=result, status_code=200 if result["ok"] else 400)


@router.post("/trimtab/reinit")
async def trimtab_reinit(raw_request: Request):
    fields = await raw_request.json()
    result = await _core(raw_request).call_utility_async("trimtab_reinit", fields)
    return JSONResponse(content=result, status_code=200 if result["ok"] else 400)


@router.get("/trimtab/knobs")
async def trimtab_get_knobs(raw_request: Request):
    return JSONResponse(content=await _core(raw_request).call_utility_async("trimtab_get_knobs"))


def attach_router(app: FastAPI):
    app.include_router(router)
'''

SCHED_ANCHOR = "    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:\n        self.current_step += 1\n"
SCHED_BODY = SCHED_ANCHOR + '''        pending = getattr(self, "_trimtab_pending_max_num_seqs", None)
        if pending is not None:  # trimtab: tighten the cap as occupancy allows
            self.max_num_running_reqs = max(pending, len(self.running))
            if len(self.running) <= pending:
                self._trimtab_pending_max_num_seqs = None
'''

INIT_ANCHOR = "    from .dev.sleep.api_router import attach_router as attach_sleep_router\n"
INIT_BODY = INIT_ANCHOR + '''
    from .dev.trimtab.api_router import attach_router as attach_trimtab_router

    attach_trimtab_router(app)
'''


def root():
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("vllm is not installed in this environment")
    return spec.submodule_search_locations[0]


def edit(path, anchor, body, check_only):
    src = open(path).read()
    if MARKER in src or "attach_trimtab_router" in src or "_trimtab_pending_max_num_seqs" in src or "trimtab_release_kv" in src:
        return "already patched"
    n = src.count(anchor)
    if n != 1:
        sys.exit(f"anchor matched {n} times in {path}, refusing to edit")
    if check_only:
        return "anchor ok"
    out = src.replace(anchor, body, 1)
    ast.parse(out)
    shutil.copyfile(path, path + ".trimtab-orig")
    open(path, "w").write(out)
    return "patched"


def main():
    check_only = "--check" in sys.argv
    r = root()
    print("core   ", edit(f"{r}/v1/engine/core.py", CORE_ANCHOR, CORE_BODY + CORE_ANCHOR, check_only))
    print("serve  ", edit(f"{r}/entrypoints/serve/__init__.py", INIT_ANCHOR, INIT_BODY, check_only))
    print("sched  ", edit(f"{r}/v1/core/sched/scheduler.py", SCHED_ANCHOR, SCHED_BODY, check_only))
    print("worker ", edit(f"{r}/v1/worker/gpu_worker.py", WORKER_ANCHOR, WORKER_BODY, check_only))
    if not check_only:
        import os
        d = f"{r}/entrypoints/serve/dev/trimtab"
        os.makedirs(d, exist_ok=True)
        open(f"{d}/__init__.py", "a").close()
        open(f"{d}/api_router.py", "w").write(ROUTER_FILE)
        ast.parse(ROUTER_FILE)
        print("router  written", f"{d}/api_router.py")


if __name__ == "__main__":
    main()
