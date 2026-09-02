"""Apply the trimtab hot-knob patch to an installed SGLang tree.

Works against the installed package rather than a source checkout, so it can
run inside any SGLang container regardless of how the image was built. Every
edit is anchor-verified. On any mismatch the file is left untouched and the
script exits nonzero. A backup of the original is written next to the target.

Usage
    python3 apply_patch.py            apply
    python3 apply_patch.py --check    report status, change nothing
"""
import ast
import importlib.util
import re
import shutil
import sys

MARKER = "_trimtab_apply_hot_knobs"

INJECT_BODY = '''
        # trimtab (github.com/numinous-technology/trimtab)
        # Hot scheduler knobs applied to the live instance. The scheduler
        # loop reads these attributes every step, so a change takes effect
        # on the next step. Values are validated against ceilings recorded
        # at boot. Handled keys are consumed here; any remaining keys fall
        # through to the stock allowlist below.
        server_args_dict = dict(server_args_dict)
        trimtab_ok, trimtab_msgs = _trimtab_apply_hot_knobs(self, server_args_dict)
        for _m in trimtab_msgs:
            logger.info(_m)
        if not server_args_dict:
            return SetInternalStateReqOutput(updated=trimtab_ok)

'''

TAIL = '''

def _trimtab_apply_hot_knobs(scheduler, args: dict):
    """Apply trimtab hot knobs to a live scheduler.

    Mutates ``args``, consuming the keys it handles, and returns
    ``(ok, messages)``. Handled knobs, all read by the scheduler loop on
    every step.

    max_running_requests   logical cap on concurrent running requests,
                           valid range 1..the value allocated at boot
    max_queued_requests    admission cap on the waiting queue, >= 0
    chunked_prefill_size   prefill chunk size for the next batch, > 0
    max_prefill_tokens     prefill token budget per batch, 1..KV pool size
    schedule_policy        fcfs, lpm, dfs-weight, lof, random, priority
    schedule_conservativeness  > 0, rebuilds the new-token-ratio watermarks
    log_level              DEBUG, INFO, WARNING, ERROR
    """
    if not hasattr(scheduler, "_trimtab_ceilings"):
        scheduler._trimtab_ceilings = {
            "max_running_requests": scheduler.max_running_requests,
        }
    ok = True
    msgs = []
    reinit = {k[7:]: args.pop(k) for k in list(args) if k.startswith("reinit.")}
    if reinit:
        r = scheduler.trimtab_reinit(reinit)
        ok = ok and r["ok"]
        msgs.append(f"trimtab reinit {r}")
    if "max_running_requests" in args:
        v = args.pop("max_running_requests")
        ceiling = scheduler._trimtab_ceilings["max_running_requests"]
        if not isinstance(v, (int, float)) or not (1 <= int(v) <= ceiling):
            ok = False
            msgs.append(
                f"trimtab rejected max_running_requests={v}, "
                f"valid range is 1..{ceiling} (the boot allocation)"
            )
        else:
            scheduler.max_running_requests = int(v)
            msgs.append(f"trimtab applied max_running_requests={int(v)}")
    if "max_queued_requests" in args:
        v = args.pop("max_queued_requests")
        if not isinstance(v, (int, float)) or int(v) < 0:
            ok = False
            msgs.append(f"trimtab rejected max_queued_requests={v}, must be >= 0")
        else:
            scheduler.max_queued_requests = int(v)
            msgs.append(f"trimtab applied max_queued_requests={int(v)}")
    if "chunked_prefill_size" in args:
        v = args.pop("chunked_prefill_size")
        if not isinstance(v, (int, float)) or int(v) <= 0:
            ok = False
            msgs.append(f"trimtab rejected chunked_prefill_size={v}, must be > 0")
        else:
            scheduler.chunked_prefill_size = int(v)
            msgs.append(f"trimtab applied chunked_prefill_size={int(v)}")
    if "max_prefill_tokens" in args:
        v = args.pop("max_prefill_tokens")
        hi = scheduler.max_total_num_tokens
        if not isinstance(v, (int, float)) or not (1 <= int(v) <= hi):
            ok = False
            msgs.append(f"trimtab rejected max_prefill_tokens={v}, valid range is 1..{hi} (the KV pool)")
        else:
            scheduler.max_prefill_tokens = int(v)
            msgs.append(f"trimtab applied max_prefill_tokens={int(v)}")
    if "schedule_policy" in args:
        v = args.pop("schedule_policy")
        try:
            new = scheduler.policy._validate_and_adjust_policy(str(v), scheduler.policy.tree_cache)
        except ValueError as e:
            ok = False
            msgs.append(f"trimtab rejected schedule_policy={v}, {e}")
        else:
            scheduler.policy.policy = new
            scheduler.schedule_policy = str(v)
            msgs.append(f"trimtab applied schedule_policy={v} (effective {new.value})")
    if "schedule_conservativeness" in args:
        v = args.pop("schedule_conservativeness")
        if not isinstance(v, (int, float)) or v <= 0:
            ok = False
            msgs.append(f"trimtab rejected schedule_conservativeness={v}, must be > 0")
        else:
            from sglang.srt.environ import envs

            t = scheduler.new_token_ratio_tracker
            t.init = min(envs.SGLANG_INIT_NEW_TOKEN_RATIO.get() * float(v), 1.0)
            t.min = min(t.init * envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(), 1.0)
            t.decay = (t.init - t.min) / envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()
            t.current = t.init
            scheduler._trimtab_conservativeness = float(v)
            msgs.append(f"trimtab applied schedule_conservativeness={v} (new_token_ratio init {t.init:.3f} min {t.min:.3f})")
    if "log_level" in args:
        v = str(args.pop("log_level")).upper()
        if v not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            ok = False
            msgs.append(f"trimtab rejected log_level={v}, must be DEBUG, INFO, WARNING or ERROR")
        else:
            import logging as _logging

            _logging.getLogger("sglang").setLevel(v)
            _logging.getLogger().setLevel(v)
            scheduler._trimtab_log_level = v
            msgs.append(f"trimtab applied log_level={v}")
    return ok, msgs
'''

READBACK_ANCHOR = '        ret["effective_max_running_requests_per_dp"] = self.max_running_requests\n'
READBACK_BODY = READBACK_ANCHOR + '''        ret["trimtab"] = {
            "max_running_requests": self.max_running_requests,
            "max_queued_requests": self.max_queued_requests,
            "chunked_prefill_size": self.chunked_prefill_size,
            "max_prefill_tokens": self.max_prefill_tokens,
            "schedule_policy": self.schedule_policy,
            "schedule_conservativeness": getattr(self, "_trimtab_conservativeness", None),
            "log_level": getattr(self, "_trimtab_log_level", None),
            "ceilings": dict(getattr(self, "_trimtab_ceilings", {}), max_prefill_tokens=self.max_total_num_tokens),
            "last_reinit": getattr(self, "_trimtab_last_reinit", None),
            "max_total_num_tokens": self.max_total_num_tokens,
        }
'''

IO_ANCHOR = "    server_args: Dict[str, Union[int, float]]\n"
IO_BODY = "    server_args: Dict[str, Union[int, float, str]]  # trimtab: str for schedule_policy, log_level\n"

REINIT_ANCHOR = "    def flush_cache(self, empty_cache: bool = True):\n"
REINIT_BODY = open(__file__.replace("apply_patch.py", "reinit_body.py")).read() + REINIT_ANCHOR

ANCHOR_RE = re.compile(
    r"(    def set_internal_state\(self, recv_req: SetInternalStateReq\):\n"
    r"        server_args_dict = recv_req\.server_args\n)"
)


def find_target():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("sglang is not installed in this environment")
    root = spec.submodule_search_locations[0]
    return f"{root}/srt/managers/scheduler.py"


def patch_io_struct(scheduler_path, check_only):
    path = scheduler_path.replace("scheduler.py", "io_struct.py")
    src = open(path).read()
    if IO_BODY in src:
        return "already patched"
    if src.count(IO_ANCHOR) != 1:
        sys.exit(f"io_struct anchor matched {src.count(IO_ANCHOR)} times, refusing to edit")
    if check_only:
        return "anchor ok"
    shutil.copyfile(path, path + ".trimtab-orig")
    open(path, "w").write(src.replace(IO_ANCHOR, IO_BODY, 1))
    return "patched"


def main():
    check_only = "--check" in sys.argv
    target = find_target()
    src = open(target).read()

    print("io_struct", patch_io_struct(target, check_only))
    if MARKER in src:
        print(f"already patched, nothing to do ({target})")
        return

    if check_only:
        n = len(ANCHOR_RE.findall(src))
        print(f"not patched, anchor matches={n} ({target})")
        sys.exit(0 if n == 1 else 2)

    matches = ANCHOR_RE.findall(src)
    if src.count(READBACK_ANCHOR) != 1:
        sys.exit(f"read-back anchor matched {src.count(READBACK_ANCHOR)} times, refusing to edit")
    if len(matches) != 1:
        sys.exit(
            f"anchor matched {len(matches)} times in {target}, refusing to edit. "
            "The installed SGLang version differs from what this patch expects."
        )

    shutil.copyfile(target, target + ".trimtab-orig")
    out = ANCHOR_RE.sub(lambda m: m.group(1) + INJECT_BODY, src, count=1)
    out = out.replace(READBACK_ANCHOR, READBACK_BODY, 1)
    if out.count(REINIT_ANCHOR) != 1:
        sys.exit(f"reinit anchor matched {out.count(REINIT_ANCHOR)} times, refusing to edit")
    out = out.replace(REINIT_ANCHOR, REINIT_BODY, 1)
    out = out.rstrip("\n") + "\n" + TAIL

    try:
        ast.parse(out)
    except SyntaxError as e:
        sys.exit(f"patched file failed to parse, original left untouched ({e})")

    open(target, "w").write(out)
    print(f"patched {target}")
    print(f"backup at {target}.trimtab-orig")


if __name__ == "__main__":
    main()
