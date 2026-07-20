# Release Training Runtime Before Final Evaluation

## Objective

Prevent full-pair checkpoint evaluation from overlapping the 2,048-world
training runtime in GPU memory.

## Scope

- `tools/train_manorl_cube1.py`
- focused trainer lifecycle tests

## Acceptance

- Preserve all result metadata before releasing the training runtime.
- Release training-only runtime/environment references before constructing the
  fresh final evaluator.
- Flush reclaimable Torch CUDA cache after garbage collection.
- Focused CPU tests and a server2 GPU3 full-scale preflight pass.
