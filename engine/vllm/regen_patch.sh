#!/bin/bash
# Regenerate the patch file from apply_patch.py against a pinned upstream vllm checkout.
# Usage: regen_patch.sh /path/to/vllm-checkout
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$1; cd "$SRC"
SHA=$(git rev-parse --short HEAD)
git checkout -q . && git clean -fdq
PYTHONPATH="$SRC" python3 "$HERE/apply_patch.py" >/dev/null
find . -name '*.trimtab-orig' -delete
git add -A && git diff --cached > "$HERE/patches/0001-hot-scheduler-knobs-$SHA.patch"
git reset -q && git checkout -q . && git clean -fdq
echo "wrote patches/0001-hot-scheduler-knobs-$SHA.patch"
