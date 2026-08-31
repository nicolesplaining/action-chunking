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

GPU 0 on this host cannot create a MuJoCo EGL device context, while the same
minimal context test succeeds on GPU 1. Policy inference remains valid on both
GPUs. All simulator clients are therefore rendered on GPU 1 and may query a
policy server on either GPU. The baseline launcher records render GPU and policy
port separately; the initial failed Spatial client completed no episodes and
was restarted from episode zero.

## 2026-08-31: exact instruction-target fixture

The public LIBERO-90 living-room tasks for alphabet soup and cream cheese share
the same region bounds, object inventory, initialization predicates, and scene.
Their BDDL files differ only in language, `obj_of_interest`, and goal. Initial
state 0 was settled for ten no-op controls, serialized, and restored in both
task environments.

The resulting pair passed byte-exact equality for both 224x224 policy images,
the 8-D robot state, the complete flattened MuJoCo state, and every object pose.
Only the prompt target differs. The fixture SHA-256 is
`c38aff06a83ff831686f817dbcd10ee6832a4ade73d69571cba3e574502aa0cf`.

## 2026-08-31: single-pair intervention pilot

This pilot is diagnostic and is not confirmatory evidence. With one saved noise
seed, the complete bidirectional 10-flow-step by 18-layer residual screen
produced 584 records. All 180 `A -> A` residual identity patches had maximum
absolute action error 0, and the all-donor flow switch reproduced the donor
endpoint with error 0. A strict cuBLAS-deterministic rerun produced byte-identical
clean traces and JSONL records.

The clean base-to-donor L2 contrasts were 0.1122 for translation, 0.0260 for
rotation, and only 0.00143 for gripper. The gripper outcome was therefore marked
ineligible rather than interpreting unstable normalized effects. At a provisional
0.8 retention threshold, translation, rotation, and target direction crossed
only after all ten source-conditioned updates. Relative to the uniform-step
null, target-direction retention was strongly late weighted: AUC 0.093 and a
final-update marginal contribution of 0.798. These numbers motivate replication
across pairs and noise seeds; they do not establish a population-level hierarchy.

Grouped action-state and velocity interventions added 120 bidirectional effects
and 60 identity controls. Every dimension identity intervention had error 0.
For this pair, replacing translation coordinates of `x_t` at the final flow step
transferred 0.467 of the translation endpoint contrast after direction
symmetrization. Single future-token follow-ups were selected from the coarse
screen without changing the confirmatory hypotheses.

Both prompts subsequently succeeded in deterministic closed-loop rollouts using
the same explicit Gaussian-noise sequence at each replan. Their first action
chunks exactly matched the offline endpoints. The alphabet-soup rollout first
contacted its target at step 61 and succeeded at step 134. The cream-cheese
rollout contacted alphabet soup at step 61 before contacting its instructed
target at step 199 and succeeding at step 249. Consequently this pair is
excluded from a clean first-contact target-identity analysis, while remaining a
valid recovery/eventual-success case. This exclusion used only clean behavior,
never patched outcomes.
