#!/bin/bash
# Regenerate the patch file from apply_patch.py against a pinned upstream checkout.
# Usage: regen_patch.sh /path/to/sglang-checkout
set -eu
SRC=$1; cd "$SRC"
SHA=$(git rev-parse --short HEAD)
git checkout -q -- python/sglang/srt/managers/scheduler.py
PYTHONPATH="$SRC/python" python3 "$(dirname "$0")/apply_patch.py" >/dev/null
rm -f python/sglang/srt/managers/scheduler.py.trimtab-orig
git diff > "$(dirname "$0")/patches/0001-hot-scheduler-knobs-$SHA.patch"
git checkout -q -- python/sglang/srt/managers/scheduler.py
echo "wrote patches/0001-hot-scheduler-knobs-$SHA.patch"
