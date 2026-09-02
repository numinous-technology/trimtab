#!/bin/bash
# Regenerate the patch file from apply_patch.py against a pinned upstream checkout.
# Usage: regen_patch.sh /path/to/sglang-checkout
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$1; cd "$SRC"
SHA=$(git rev-parse --short HEAD)
git checkout -q -- python/sglang/srt/managers/scheduler.py python/sglang/srt/managers/io_struct.py
PYTHONPATH="$SRC/python" python3 "$HERE/apply_patch.py" >/dev/null
rm -f python/sglang/srt/managers/*.trimtab-orig
git diff > "$HERE/patches/0001-hot-scheduler-knobs-$SHA.patch"
git checkout -q -- python/sglang/srt/managers/scheduler.py python/sglang/srt/managers/io_struct.py
echo "wrote patches/0001-hot-scheduler-knobs-$SHA.patch"
