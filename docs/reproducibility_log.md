# Reproducibility log

This log records environment validation and discrepancies encountered while
executing the pinned public implementation. Raw machine-readable outputs remain
in the experiment artifact store and are summarized here at coherent milestones.

## 2026-08-31: reference environment

- Compute: 2x NVIDIA H100 80 GB HBM3 with NVLink.
- Driver: 580.105.08.
- OpenPI: `215abfb217dbac7d5f1273282331b9b1866c0479`.
- LIBERO: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
- Python: 3.11.16 managed by uv 0.12.7.
- PyTorch: 2.7.1+cu126; both H100s visible.
- JAX: 0.5.3; both H100s visible.
- OpenPI Transformers compatibility check: passed.
- Upstream `pi0_test.py` and `model_test.py`: 9 passed. The 1,699 emitted
  warnings are upstream JAX/Flax deprecations and are retained in the raw log.

The resolved `pi05_libero` configuration has action horizon 10, padded action
dimension 32, Gemma 2B VLM configuration, and Gemma 300M action expert with 18
layers. LIBERO returns the first seven physical action dimensions.

## 2026-08-31: checkpoint conversion

The official public checkpoint was downloaded from
`gs://openpi-assets/checkpoints/pi05_libero` (11.6 GiB) and converted with the
pinned OpenPI `convert_jax_model_to_pytorch.py` script.

The converter did not copy normalization assets for this directory layout: it
looked for `assets` beside the checkpoint directory rather than inside it. The
official checkpoint's unmodified `assets/physical-intelligence/libero` tree was
therefore copied into the converted checkpoint. Upstream source was not edited.

On an identical deterministic image pair, state, prompt, and 10x32 Gaussian
noise tensor:

| comparison | result |
| --- | ---: |
| max absolute action error | 0.0071143 |
| mean absolute action error | 0.0017788 |
| RMSE | 0.0024182 |
| cosine similarity | 0.9999892 |

The public output transform was applied before comparison, yielding a 10x7
physical action chunk.

## 2026-08-31: intervention adapter

The explicit flow sampler matches `PI0Pytorch.sample_actions` exactly on the
fixed conversion fixture (maximum absolute error 0). Enabling trace-only hooks
also changes the output by exactly 0 and captures all 180 expected sites: 10
flow steps by 18 post-layer action-expert residuals.

Unit tests validate all-position and selected-position patches as well as
preservation of non-action suffix tokens. Four local instrumentation tests pass.

## 2026-08-31: LIBERO container discrepancy

The pinned upstream LIBERO Dockerfile failed while source-building `gym==0.25.2`
under Python 3.8. Its isolated build selected a newer setuptools release whose
runtime code uses syntax unavailable in Python 3.8. The derived
`docker/libero-client.Dockerfile` preserves upstream runtime requirements and
base-image digest while constraining only isolated build setuptools to 69.5.1.
This change must be included in the final environment limitations and artifact
manifest.

The derived image built successfully with all 65 upstream runtime package
versions unchanged. A one-trial-per-task LIBERO Spatial smoke evaluation then
completed 10/10 tasks successfully. The first episode took 336 seconds because
of one-time `torch.compile` autotuning; the remaining episodes took 15--23
seconds each. LIBERO emitted `EGL_NOT_INITIALIZED` only while destroying the
already-completed render context at process exit; all ten videos and success
records had already been written.
