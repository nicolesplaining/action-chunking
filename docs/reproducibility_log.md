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

## 2026-08-31: official LIBERO baselines

The full pinned OpenPI evaluation protocol completed 50 trials for each of the
10 `libero_object` tasks. Pi0.5 succeeded in 491/500 episodes (98.2%), exactly
matching the aggregate result reported in the pinned OpenPI LIBERO README.
Per-task rates ranged from 96% to 100%. The raw evaluator log was parsed only
after all 500 success records were present; the versioned parser rejects empty
or incomplete expected episode counts.

The corresponding `libero_spatial` run succeeded in 493/500 episodes (98.6%),
one episode below the pinned OpenPI README's 98.8%. Per-task rates again ranged
from 96% to 100%. Aggregate and per-task tables report Wilson 95% binomial
intervals; the published value lies well within run-to-run sampling uncertainty.

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

## 2026-08-31: clean-only target-pair expansion and exclusion audit

All six prompt contrasts among four objects in the same public LIBERO-90 living
room scene were generated for 16 initialization states and four shared noise
seeds. The 96 fixtures collapse to 48 independent serialized simulator states,
which are retained as the outer dependency cluster. Only three fixtures passed
the provisional all-seed direct-target-direction screen; all were cream-cheese
to ketchup contrasts at initialization states 0, 8, and 14.

The three survivors were then evaluated closed loop for all four noise seeds.
Every one of the 24 first chunks reproduced its offline clean endpoint exactly,
but zero of 12 paired seed jobs succeeded on both prompts. All 12 cream-cheese
runs succeeded and all 12 ketchup runs failed. Zero of 24 side rollouts first
contacted its instructed task object. These pairs are therefore excluded from
target-identity causal analysis. The failed result also falsifies the use of an
initial five-action Cartesian direction sign as a categorical target-identity
gate; the raw screen remains preserved as a negative pilot result.

An audit of the pinned public BDDL task registry found 44 strict
manipulated-object substitutions and three strict destination substitutions in
LIBERO-90. Across the four official evaluation suites, `libero_goal` contains
one strict manipulated-object pair and four strict destination pairs; the other
official suites contain no exact prompt-only target substitution. For the 16
states of the official goal wine-bottle versus bowl pair, clean translation
contrasts were large under all seeds (minimum per-state values 1.24--1.92), but
zero states passed the direct-target sign criterion because both prompts'
initial actions project toward the bowl direction. This supports replacing the
invalid direct-line proxy with phase-aligned clean trajectory and contact
outcomes, not relaxing its threshold after observing interventions.

## 2026-08-31: first behaviorally valid official-suite causal pair

Official `libero_goal` initialization state 0 provides a strict wine-bottle
versus bowl target substitution with the destination fixed. Under shared saved
noise, both closed-loop endpoints reproduced their offline first chunks exactly,
contacted the instructed object first (steps 28 and 29), and succeeded at step
81. The initial chunks contained no closure command, so gripper-closure outcomes
are right-censored and excluded even though their continuous gripper vectors
have a small nonzero L2 difference.

All 180 residual identity patches, 60 grouped-dimension identity patches, and
the full-donor flow switch had exactly zero action error. At the provisional
20% formation-error tolerance, paired translation, rotation, and all-action
contrasts were already aligned with their final contrasts at the first flow
step; the direct-target projection reached tolerance at step 6. Causal suffix
switches told a different story: translation, rotation, all-action, and target-
direction retention did not cross 0.8 until step 9 of 10. Thus a paired action
contrast can be legible in the first clean estimate while remaining causally
editable until the penultimate integration update.

At the final flow step, grouped `x_t` translation interchange transferred 0.571
of the symmetric translation contrast and grouped rotation interchange
transferred 0.937 of rotation. The strongest positive all-position residual
transfer was 0.094 at the final action-expert layer. Single-token patches showed
exactly diagonal output influence at the final layer, as expected from the
token-wise output head, but earlier layers propagated effects off diagonal
through subsequent action-token mixing. These are single-pair pilot effects,
not population estimates.

The unchanged Event-SAE/AWE position-only extractor, pinned at the revisions in
`docs/sources.md`, found paired waypoints `[22, 61, 80]` and `[23, 63, 80]` at
its public 0.05 error threshold. A 0.01--0.075 sensitivity sweep preserved an
early waypoint near the observed first-contact event and a later transport/
placement waypoint. Event-SAE's geometric-gripper mode selected every step
because OpenPI emits continuous gripper commands while the public toggle helper
tests exact inequality; it is therefore not used for phase labels.

Finally, flat MuJoCo state alone did not reproduce observations for initialization
indices after zero. Fixture generation advances a seeded environment reset once
per index, and some observation-relevant task state lies outside the flat state.
Replaying that reset sequence before state restoration recovered strict image,
proprioception, simulator-state, and first-chunk equality for state 1; the full
strict state audit uses this procedure. The completed 16-state audit reproduced
all three model inputs, the simulator state, and the clean first chunk exactly
for all 32 side rollouts. Every side first contacted its instructed target. The
two sides both completed the task in 15/16 paired states (Wilson 95% interval
0.717--0.989); state 3's wine-bottle side contacted the correct target but later
failed the task. The dual-success primary pilot set therefore has 15 states,
while all 16 enter the explicitly labeled contact-valid sensitivity analysis.

The online intervention server then reproduced both offline clean endpoints and
all source-identity, donor-endpoint, residual-identity, and dimension-identity
controls with zero maximum absolute error. Representative bidirectional
nonidentity residual and `x_t` translation patches also matched their offline
outputs exactly. In the state-0 closed-loop positive
control, applying the full donor condition at every replan symmetrically swapped
first contact: the wine-bottle instruction contacted the bowl at step 29 and the
bowl instruction contacted the wine bottle at step 28. Neither side contacted
its source target or completed the source task. The full-source identity control
reproduced the original target contacts and both 81-step successes. This is a
causal target-selection positive control, but remains a single-state pilot until
the clean-selected state sweep is complete.

The complete state-0 first-contact sweep was monotonic and symmetric. Boundaries
0--7 transferred donor identity in both directions; boundaries 8--10 retained
source identity in both directions. The categorical commitment step was
therefore 8/10 for both directional curves (source-retention AUC 0.25). This is
earlier than the provisional continuous-action 0.8-retention boundary at step 9,
showing why categorical contact and geometric action-distance endpoints must be
reported separately.

The prespecified seed-0 offline sweep then completed on all 15 dual-success
scene-state clusters. A 10,000-replicate scene-state cluster bootstrap confirmed
that the paired full-action, translation, and rotation contrasts were formed at
flow step 0 (all 95% intervals 0--0). Target-direction formation was also step
0, with a wider 0--3 interval. In contrast, 0.8 source retention was not reached
until step 10 for the full action, translation, and target direction (all 95%
intervals 9--10), and step 9 for rotation (95% interval 9--9). The corresponding
retention AUCs were 0.232, 0.220, 0.159, and 0.310. Gripper timing remained
ineligible because neither clean initial chunk contained a closure event. This
is the first multi-state positive result: the output contrast is geometrically
legible after the first flow update while remaining causally editable until the
last one or two updates. It remains a single instruction contrast and a single
noise seed; the closed-loop state sweep and additional shared-noise seeds are
required before generalization.
