# Upstream pull requests

Ready-to-push patches for both engines, generated from the same patchers the
repo ships. Each is one commit against the pinned upstream SHA in the file
name of engine/<engine>/patches, with the measured numbers in the message.

To open them.

```
cd /path/to/sglang && git am /path/to/trimtab/docs/upstream/0001-Feature-Extend-set_internal_state-*.patch
cd /path/to/vllm   && git am /path/to/trimtab/docs/upstream/0001-Core-Hot-scheduler-knobs-*.patch
```

Then push to a fork and open the PR. The commit message is the PR body.
Expect upstream to ask for the trimtab prefix to be dropped from names,
which is fine, the adapters read the manifest and can follow a rename.

## State, 25 September 2026

Both PRs were rebased onto current upstream main, formatted to each project's
style, given unit tests, and re-described.

| | sglang #37661 | vllm #55018 |
|---|---|---|
| contents | knob widening only | knobs plus warm reinit |
| commits | 1 | 3 |
| tests | `test/registered/unit/managers/test_trimtab_hot_knobs.py` | `tests/v1/engine/test_trimtab_knobs.py` |
| mergeable | yes | yes |
| blocked on | the `run-ci` label, which only listed contributors can add | the `ready` label, or four merged vLLM PRs |

The SGLang KV pool reinit was split out of #37661 and is kept in
`sglang-followup-reinit.py.txt` for a follow-up PR once the knobs land. The
three review findings on the vLLM PR are addressed in its second commit:
validation before release, shutting down the replaced scheduler, and handling a
cache that is not in a CuMem pool.

Neither project runs CI on a first-time contributor's PR without a maintainer
acting first, so the remaining step in both is a person, not a patch.
