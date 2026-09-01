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
    """
    if not hasattr(scheduler, "_trimtab_ceilings"):
        scheduler._trimtab_ceilings = {
            "max_running_requests": scheduler.max_running_requests,
        }
    ok = True
    msgs = []
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
    return ok, msgs
'''

READBACK_ANCHOR = '        ret["effective_max_running_requests_per_dp"] = self.max_running_requests\n'
READBACK_BODY = READBACK_ANCHOR + '''        ret["trimtab"] = {
            "max_running_requests": self.max_running_requests,
            "max_queued_requests": self.max_queued_requests,
            "chunked_prefill_size": self.chunked_prefill_size,
            "ceilings": getattr(self, "_trimtab_ceilings", {}),
        }
'''

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


def main():
    check_only = "--check" in sys.argv
    target = find_target()
    src = open(target).read()

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
