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
