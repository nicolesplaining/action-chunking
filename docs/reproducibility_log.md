# Reproducibility log

## 2026-09-01: public-catalog obstacle expansion frozen

Protocol version 0.14 was frozen after the wine/bowl obstacle family exhausted
all 144 geometry checks and before generating another object-family obstacle
fixture. Every rejection was a pre-policy target--obstacle bounding-sphere
overlap; no clean or intervention outcome existed. The expansion reuses the
previously hashed public retarget-screening rows and exclusions, but serializes
a task-diverse round-robin order `(init_index, first-occurrence target-pair
rank)` so early screening covers different object geometries. The nine-point
placement grid, geometry exclusions, clean eligibility thresholds, and
first-clean-pass stop rule are unchanged. Patched and dynamic outcomes remain
sealed during selection.

## 2026-09-01: obstacle scene-order expansion frozen after geometric exhaustion

Protocol version 0.13 was frozen after initialization state 0 produced no
geometrically valid placement and before any obstacle policy call or
intervention outcome. All nine state-0 grid points were rejected solely by the
already frozen MuJoCo overlap and clearance checks. The search now scans the
same 16 exact target-pair states in original manifest order and retains the
existing nested placement order and every exclusion. It stops at the first
clean-eligible placement using no patched or dynamic outcome. No geometry or
behavioral threshold was changed.

## 2026-09-01: late visual-safety update pilot frozen

Protocol version 0.12 was frozen before inspecting any obstacle-pose clean or
intervention outcome. The robot will execute in the moved-obstacle state while
the first flow trajectory is initialized from the exactly paired original-pose
image and switched to the live moved-obstacle condition at every boundary
`0..10`. The full-source chunk must induce obstacle contact within the first
five controls, whereas full restart must avoid contact and eventually complete
the unchanged task. A continued boundary is successful only if it avoids the
obstacle through the first chunk and eventually succeeds, with exact fixture,
simulator, boundary-zero, and neural-function-evaluation controls. All
boundaries are reported; the single selected scene is descriptive and cannot
support a population timing-prediction claim. This amendment was motivated by
the construct-validity concern that successful instruction redirection is good
instruction following, not evidence that a policy was previously undecided.
The implementation runs only restart and the fully old-conditioned `k=10`
endpoint first; intermediate boundaries remain sealed unless this gate passes.

## 2026-09-01: chunk-boundary recovery decomposition frozen

Protocol version 0.11 was frozen during transition-exact candidate generation,
before the corrected endpoint gate, selected action-only predictions, or any
corrected continuation output existed. Every dynamic-retarget fork now fails
closed unless its registered controller replay is exact; checking the terminal
MuJoCo state alone is no longer accepted.

The secondary recovery decomposition records old- and new-target contact within
the five executed actions before the first clean replan, absence of registered
contact in that chunk, first-contact replan index, correct-target/task success
that occurs only after subsequent clean replanning, and eventual task success.
The registered U1 composite remains unchanged. These fields will distinguish an
immediate unsafe chunk, later wrong-target failures at chunk boundaries, and
successful clean-replanning rescue without reinterpreting late redirection as
model indecision. The implementation and summary tests passed before outcomes.

## 2026-09-01: matched pi0 training completed; final identity frozen

The matched public `pi0_libero` run completed all 30,000 optimizer updates with
the registered global batch size 32, two-H100 FSDP, 0.99 EMA, seed 42 experiment
name, pinned OpenPI revision, and frozen normalization statistics. The last
reported 100-update block at loop index 29,900 had finite loss 0.0108, gradient
norm 0.1366, and parameter norm 1383.1084. These training diagnostics are not a
competence result.

Source audit before competence evaluation found that OpenPI saves the final
30,000-update state under zero-based loop label `29999`, not `30000`: the train
loop ranges over `0..29999`, increments `train_state.step` inside every update,
and passes the loop index to Orbax. The checkpoint rule remains “final update
only”; this is a correction of its filesystem label, not checkpoint selection.
The finalized required artifact hashes are `_CHECKPOINT_METADATA`
`cf454e8412c1e734cec099baf1a0638b56ad07b9e70219e17ea0a8b780d5fae8`,
parameter manifest `1a338f26792c1614a315c0e9bac7bb532c21f9bc2ac6b196be0fcad862fe0894`,
and training-state manifest
`f77c5744e270d94646d1fde33e2f4ff29fabf801b4ff5b6eefce9a77a02acfe1`.
All suite, pair, and conversion launchers now call one tested validator that
requires the experiment name, label, committed Orbax metadata, and these hashes.
No pi0 competence or intervention outcome existed when this correction was made.

## 2026-09-01: partial recovery screen rejected; controller replay frozen

Protocol version 0.10 was frozen before any corrected recovery endpoint or
continuation outcome. The partial version-3 endpoint screen is rejected in full
and retained only as an audit artifact. It had registered byte-exact policy
inputs, Gaussian draw, source chunk, and MuJoCo flat state, yet a failed fork's
end-effector path diverged from its source clean path by centimeters within the
five-action horizon. For example, after the first action from state 16's base
snapshot, source and fork end-effector positions were approximately
`[-0.1901, -0.0606, 1.0356]` and `[-0.1816, -0.0558, 1.0436]`. The associated
old-target event was therefore not a valid replay test.

The pinned public stack uses robosuite 1.4.1. Source inspection confirmed that
the operational-space controller and its interpolators maintain state outside
MuJoCo's flattened simulator state. The corrected fork reproduces the clean
seeded reset and registered-initial-state restoration, then deterministically
replays every clean action to the registered replan boundary. It registers
hashes for the replay actions and full
simulator-state prefix and requires array equality at every physical step in
both task environments. Downstream utility validation now fails closed unless
this controller-replay gate is true. The invalid screen and its automatic
handoff were stopped without deleting their outputs; no partial row informed
candidate selection, U1--U3 outcomes, or thresholds.

The first corrected validation reran the previously divergent state-16 base
snapshot. Both task forks reproduced all 41 registered simulator states with
zero maximum absolute error; all three live model-input arrays were also exact,
and the old-condition action chunk matched the registered clean chunk digest.
The old wine-bottle contact reappeared at step 4, exactly matching the source
clean event. The new-instruction endpoint contacted the old target at step 3,
so this state fails the preregistered restart-avoidance gate and is not a utility
success. This is a positive construct-validation result only.

## 2026-09-01: obstacle-pose pilot frozen before clean or patched outcomes

Protocol version 0.9 adds a same-task obstacle family before generating any
obstacle-pose fixture or rollout. It reuses the public LIBERO simulator and the
existing exact target-pair scene rather than introducing a synthetic renderer.
The unchanged instructed object is the target and the alternate registered
object is a task-irrelevant distractor. Only the distractor free joint's planar
coordinates may differ; configuration validation rejects any change to robot
state, target pose, other generalized positions, velocity, actuator state,
obstacle height, or orientation.

The frozen placement grid uses path fractions 0.35, 0.50, and 0.65 and lateral
offsets 0, -5 cm, and +5 cm in nested manifest order. MuJoCo geometry bounds
provide pre-policy overlap exclusions. The clean-only screen requires exact
target-first dual success, no first-chunk obstacle contact, counterfactual
intersection of the original trajectory with the moved-obstacle corridor,
clearance beyond the obstacle bound, at least 1.5 cm clearance improvement, and
at least 1 cm first-horizon endpoint change. The first passing candidate is
selected; all rows remain in the denominator. No obstacle clean or intervention
outcome existed when these rules and tests were added.

## 2026-09-01: matched pi0 intervention grid frozen before control outcomes

Protocol version 0.8 was frozen while the public pi0 training run was still at
approximately 25,000/30,000 updates. No step-30,000 competence, conversion-
parity, or pi0 intervention outcome existed. The earlier competence and parity
gates remain unchanged and fail closed.

Passing pi0 is now automatically evaluated on the clean-eligible intersection
with pi0.5. The coarse grid uses all ten flow steps, all 18 action-expert layers,
joint action-position residual patches, and the existing dimension groups. The
position grid uses steps 0, 7, 8, and 9; layers 0, 8, 14, and 17; and all 50
native pi0 action positions. The analyzer was amended before this grid to retain
both directional retention curves and their asymmetry, and to treat positions
0--9 as a primary window even when the native model exposes additional
positions. All native positions remain in the released heatmap.

The paired comparison code was fixed before outcomes. It matches serialized
scene states, reports timing and flow-shape differences, compares common
residual and action-dimension cells, compares positions 0--9 directly, and
compares ten normalized chunk-time bins (`1` pi0.5 position versus `5` pi0
positions per bin). Scene bootstraps, exact sign-flip tests for at most 20
clusters, and within-metric BH correction are computed from the common-state
units. This preserves the model-level-control interpretation and does not claim
that pretraining, horizon, and architecture have been isolated.

## 2026-09-01: independent-cluster utility analysis frozen before outcomes

Protocol version 0.7 was frozen while exact replan-aligned candidate generation
for the corrected 16-state block was still running. No corrected endpoint-gate,
action-only prediction, or continuation-sweep outcome existed or was inspected
at this amendment. The clean prerequisite had completed 16/16 exact-state jobs
with 15/16 dual successes; candidate generation was an outcome-blind prefix and
the endpoint gate remained absent.

The utility runner now selects the first endpoint-eligible direction in frozen
gate order within each independent scene cluster. Additional directions remain
auditable but cannot inflate the primary denominator. Pair and side are both
part of prediction and rollout paths, preventing two eligible directions from
colliding. The catalog screen now always follows the pilot rather than running
only after a zero-eligibility pilot. Once its frozen stop rule is reached or the
catalog is exhausted, a hashed handoff gathers the endpoint gates and candidate
manifests, freezes every selected action-only prediction, and only then permits
continuation outcomes.

The registered U1 summary is exact and within-one-boundary accuracy, mean
absolute error, Spearman correlation, success at the predicted boundary, and
failure at the immediately following boundary. U2 uses a composite of
new-target-first contact and eventual new-task success, the one-sided exact
Clopper--Pearson upper bound on paired boundary-seven losses, exact 3-versus-10
post-event velocity-evaluation accounting, and a seed-zero 10,000-resample
scene-cluster bootstrap interval for median paired latency savings. These
statistics and their implementation tests were added before any construct-valid
utility outcome.

## 2026-09-01: pi0 conversion-parity gate frozen

Before the step-30,000 pi0 checkpoint or any pi0 intervention outcome existed,
the post-competence JAX-to-PyTorch handoff was made numerical. Conversion uses
OpenPI's pinned public `convert_jax_model_to_pytorch.py` unchanged and can run
only after both competence levels pass. Parity then uses both directions of all
16 held-out target fixtures and identical seed-zero model-native `50 x 32`
noise. Each of the 32 physical-action tensors must have maximum absolute error
at most 0.02 and cosine similarity at least 0.999. The earlier pi0.5 synthetic
conversion check had maximum error 0.00711, but its permissive 0.15 threshold is
not reused. Any failed pi0 case blocks activation-level architecture claims;
the JAX behavioral competence result remains separately reportable.

## 2026-09-01: population action-token grid frozen

Before any population target-token intervention was generated, the missing
action-position axis was frozen on the same 15 clean-eligible wine/bowl scene
states used for the target population analysis. The grid crosses flow steps
`0,7,8,9`, action-expert layers `0,8,14,17`, and all ten pi0.5 future action
positions. The primary site is flow step 9, layer 17; its preregistered grouped
contrast is mean symmetric NCTE over positions 0--4, which are executed before
replanning, minus positions 5--9. Inference remains clustered by scene state.
The full 160 cells per eligible property are retained with within-property BH
adjustment, while the grouped primary contrast uses a scene bootstrap interval
and exact sign-flip test. This replaces no existing layer or dimension result
and prevents choosing token groups from the eventual heatmap.

## 2026-08-31: public-catalog expansion order frozen

Before the corrected 16-state endpoint screen completed, strict task-pair
catalogs were regenerated from the pinned public LIBERO registry. LIBERO-90
contains 44 manipulated-object and three destination pair definitions; the
catalog SHA-256 is
`7b7da0b386367a1c0a5876d7df4ac05a8ce6086c59d54ebeaab00ff2d4a15ecc`.
LIBERO-goal contains one manipulated-object and four destination definitions;
its catalog SHA-256 is
`bc02e792876dea72be6d4817f41e93d5ee8ed60771bc170b423f0adb509080cb`.

Protocol version 0.6 freezes the utility-screen expansion artifact at
`catalogs/retarget_screening_plan.json`, SHA-256
`a6bd9fc6d72a31b111fbcb7ebbd81e43cb80068c1982f759047e9f42ff142762`.
It orders 45 manipulated-object pair definitions by canonical-scene hash,
suite, base task ID, donor task ID, and initialization index. After explicit
exclusion of wine-bottle/bowl indices 0--31 used in prior pilots, it contains
2,218 rows spanning 718 unique `(suite, scene, initialization)` clusters. The
screen stops only after at least 59 unique eligible clusters have been frozen,
or after catalog exhaustion. Directions and repeated target contrasts never
increase the independent cluster count. Selection uses endpoint eligibility
only; no intermediate continuation outcome existed when this plan was written.

## 2026-08-31: exact clean-replan input correction

The first protocol-0.4 aligned candidate was screened using endpoint controls
only. Its simulator state, fixture-relative live inputs, and indexed seed-zero
noise were exact, but its old-condition first chunk differed from saved clean
chunk 8 (maximum absolute action difference 0.00913). It was therefore rejected
by the source-chunk identity gate before competence or any intermediate
continuation boundary was run. This was a construct-control failure, not a
negative recovery outcome.

Inspection showed that the candidate fixture contained observations regenerated
from the saved MuJoCo flat state rather than the exact observations used by the
clean policy at that replan. Equality between a newly generated fixture and a
new live regeneration cannot establish equality to the historical clean policy
input. Protocol version 0.5 therefore records exact external image, wrist image,
and proprioception arrays at every clean replan. A valid fork uses these arrays
for its first policy call, restores the saved simulator state for execution,
advances to the matching clean noise draw, verifies registered input hashes,
and reproduces the saved old action chunk byte-for-byte. Later replans use live
observations normally.

The protocol-0.4 candidate generator was stopped after this failure; its partial
fixtures and one endpoint-only screen are retained as rejected diagnostics. No
intermediate continuation outcome was produced. Clean trajectories are rerun
to create versioned exact-replan input traces before aligned screening resumes.

## 2026-08-31: replan-alignment correction before held-out interventions

Before any held-out intermediate-boundary continuation was generated, an audit
found that an arbitrary pre-contact environment state does not necessarily
coincide with a policy replan and that resetting the seed-zero generator at
that state reuses noise draw zero rather than the clean trajectory's draw for
that replan. Those forks can therefore sample a different old-condition chunk
and do not isolate editability of the clean in-progress plan.

Protocol version 0.4 corrects the fork. For each clean trajectory it selects the
latest five-action replan boundary strictly before first instructed-target
contact, records its clean replan index, advances the identical Gaussian-noise
sequence to that index, and hashes the corresponding saved clean action chunk.
Eligibility now additionally requires the bounded old-condition endpoint's
first chunk to match that hash exactly. Prediction and dynamic-retargeting runs
consume the same indexed noise draw. Unit tests cover boundary selection, RNG
advancement, and saved-chunk lookup.

The earlier state-zero offsets-one-through-five screen is retained only as a
superseded diagnostic. It generated no intermediate continuation outcomes and
cannot exclude those scene states under the corrected gate. The corrected
held-out clean-competence run itself contains no retargeting outcomes. Replan-
aligned candidate generation, endpoint eligibility, frozen action-only
predictions, and only then closed-loop continuation occur in that order.

## 2026-08-31: construct-validity amendment and utility registration

An internal construct-validity review identified that the suffix-switch result
had been overinterpreted. Successful redirection under later donor conditioning
demonstrates retained causal control by the instruction counterfactual; it does
not demonstrate that the policy was undecided, lacked an internal plan, or had
not formed a persistent representation. The numerical retention curves,
interventions, thresholds, and raw outcomes are unchanged. Protocol version 0.2
renames the primary estimand *conditional editability* and records the stronger
nonclaim before any inference-utility outcome is inspected.

The amendment registers a consequential test. A new instruction arrives after
`k` old-instruction velocity-field evaluations. Continuation retains `x_k` and
uses only the remaining updates; restart discards `x_k` and regenerates from the
same original noise. The new instruction defines the evaluated task and every
later clean replan. Boundaries 0, 7, 8, 9, and 10, a five-percentage-point paired
noninferiority margin, target-first contact, eventual new-task success,
completion steps, synchronized post-event latency, and velocity-evaluation
counts were frozen before running held-out episodes. Boundary 7 is the primary
safe-point prediction and requires three post-event velocity evaluations versus
ten for restart.

The sampling adapter now supports exact partial integration and resumption from
an explicit action state. A unit control proves that split prefix/suffix
integration is bitwise identical to the original one-pass conditioning switch.
The inference server separately times new-condition preparation and the
post-event integration, and its restart control deliberately computes then
discards the old prefix. Boundary-zero continuation and restart actions, restart
actions across event boundaries, initial observations, simulator state, and
first-replan-only application are mandatory exact controls. No dynamic-
retargeting behavioral or latency result had been generated when this amendment
was written.

## 2026-08-31: dynamic-retargeting latency and behavioral pilot

The frozen dynamic-retargeting adapter was evaluated on the original state-zero
wine-bottle-versus-bowl pair with shared noise seed zero. This state is part of
the mechanistic pilot set and is not held-out confirmatory evidence. A latency
microbenchmark used both directions, one warmup, and five measured repetitions
per strategy and boundary. All restart actions were exactly invariant to the
simulated event boundary, and boundary-zero continuation was action-identical
to restart. At boundary 7, continuation used three rather than ten post-event
velocity evaluations and reduced median synchronized post-event time from
378.95 ms to 191.67 ms, a 49.0% reduction. New-condition preparation was
included in both measurements and accounts for the fixed latency floor.

The closed-loop pilot then treated the new instruction as the actual task,
applied retargeting only at the first replan, and used the new instruction for
all later clean replans. Exact initial observations and simulator states were
restored in every run. Restart reached the new object first and completed the
new task in both directions. Boundary-7 continuation also achieved 2/2
new-target-first contacts and 2/2 task successes with three post-event velocity
evaluations; mean post-event time was 179.69 ms versus 400.84 ms for restart, a
55.2% reduction. Completion took 76 and 72 environment steps versus 81 and 81
for restart. This is a positive inference-efficiency pilot, not a population
noninferiority result.

Crucially, continuation at boundaries 8, 9, and 10 also achieved 2/2
new-target-first contacts and 2/2 eventual successes. Boundary 10 performs no
new-instruction velocity evaluation, so its success shows that the first five
executed actions from the old-conditioned chunk were behaviorally benign and
later clean replanning was sufficient. The initial-state design therefore does
not establish that the measured editability boundary predicts the last
successful correction point. Treat this as a sensitivity failure for U1 and
U3, not as evidence that all late samples remain editable.

Before confirmatory utility outcomes are inspected, add an episode-level
failure-induction gate: at the intervention state, the fully old-conditioned
chunk must produce the registered old-target or safety event, while a clean
restart under the new instruction must avoid that event and remain task
competent. Candidate states will be screened near pre-contact decision points;
screening outcomes and excluded states will be retained. The five-action
execution horizon remains frozen for the primary receding-horizon comparison,
with longer execution horizons reported only as sensitivity analyses.

The superseded first failure-induction diagnostic used the saved state-zero clean
trajectories and offsets one through five steps before each original contact.
The bounded event screen evaluated ten origin directions using only the fully
old-conditioned endpoint and a clean new-instruction restart. Zero of ten
passed. At offset one, both directions induced the old event, but restart also
contacted the old target within five actions. Across offsets two through five,
the donor-origin old controls contacted the bowl within the horizon, but restart
did so as well; several base-origin old controls did not induce wine-bottle
contact under the shared event noise. No intermediate continuation outcome was
generated for these candidates. Because those snapshots were not constrained
to replan boundaries and their RNG restarted at draw zero, this block is not a
valid eligibility screen and makes no exclusion decision. Fresh initial-state
indices 16--31 were generated as the first sealed held-out screening pool for
the corrected procedure.

The first nonzero-index fixture batch was rejected before inference because
strict restoration found unequal external and wrist images at index 16. The
generator had selected simulator initial states 16--31 but had not advanced the
environment's renderer reset sequence past indices 0--15. Both task-local
environments now explicitly replay `start_index` resets before collecting a
nonzero range. A regression test covers this reset count. The rejected batch is
retained, and the corrected batch uses a new versioned output directory; no
rollout or continuation outcome from the invalid fixtures was generated.

## 2026-08-31: official pi0.5-LIBERO baseline replication

The unchanged official OpenPI evaluator completed 500 episodes per suite for
all four suites. Spatial achieved 493/500 = 98.6% (Wilson 95% interval
0.971--0.993), object 491/500 = 98.2% (0.966--0.991), and goal 490/500 = 98.0%
(0.964--0.989). The public pinned OpenPI table reports 98.8%, 98.2%, and 98.0%
respectively, so object and goal match exactly and spatial differs by 0.2
percentage points. LIBERO-10 achieved 465/500 = 93.0% (0.904--0.949), 0.6
percentage points above the published 92.4%. Across exactly 2,000 episodes the
four-suite total is 1,939 successes, for both a micro-average and an equally
weighted suite macro-average of 96.95% (micro Wilson interval 0.961--0.976),
matching the published four-suite average of 96.85% within 0.1 percentage
points. Task-level tables, strict summaries, and raw logs are retained.

The matched public pi0 base checkpoint was converted with the pinned OpenPI
converter. Because the public pi0 checkpoint does not provide LIBERO
normalization statistics, the unchanged public `pi0_libero` data configuration
was run over all 8,545 batches of `physical-intelligence/libero`. The resulting
`norm_stats.json` is 1,951 bytes with SHA-256
`f68a5fafe15e1577b7bb2c6fc4837a7d1669e2e9be3752f2589c3d327c6f8ccf`.
The access credential was supplied only to the downloader process and was not
written to the repository, shell history, or experiment metadata. These frozen
statistics are the input to the preregistered two-device pi0 control training.

## 2026-08-31: matched pi0 control smoke

The exact upstream `pi0_libero` full-finetuning recipe passed a two-update smoke
test with batch size 32 and two-device FSDP. The launcher verified the pinned
OpenPI revision, the frozen normalization-statistics hash, and a complete public
LeRobot cache containing 1,693 parquet files. LeRobot indexed 273,465 examples,
the official public pi0 base parameters restored without error, and both H100s
held an approximately 60.7 GiB JAX allocation during the update.

The first reported update had finite metrics: loss 0.1812, gradient norm 3.7253,
and parameter norm 1377.8652. Both requested updates completed in 55 seconds
after initialization, including first-step compilation, and Orbax checkpoint 1
finalized without an asynchronous-save error. This establishes that the matched
training pipeline is executable; it is not a policy-performance result.

At training step approximately 16,800, before any pi0 evaluation or
intervention outcome existed, the competence gate was made numerical. The
finalized step-30,000 policy must achieve at least 90% success over the unchanged
500-episode `libero_goal` evaluation with a Wilson 95% lower bound of at least
87%, and at least 12/16 dual-successful exact held-out target-pair controls at
noise seed zero. Model contrasts use only the clean-eligible intersection. A
failed gate is reported as competence limitation; thresholds and checkpoint
selection are not revised after evaluation.

Before any held-out intermediate continuation boundary was run, U1's predictor
was also made executable and directional. It uses the exact eligible fixture,
shared noise, offline old-to-new flow switch, target-direction affinity contrast
of at least 0.01, and the existing 0.8 isotonic retention threshold. If the
editability boundary is `k*`, the frozen prediction is that `k* - 1` is the last
successful no-restart correction boundary. The prediction file must predate the
corresponding closed-loop boundary outputs.

A pre-outcome power audit then showed that the 16-state held-out block cannot
establish the registered five-point noninferiority margin. With zero paired
continuation losses, the exact one-sided 95% upper loss bound is below 0.05 only
at 59 or more independent eligible scene clusters. Directions, seeds, and
tokens remain repeated measures and are not used to inflate this count. The
16-state block is retained as an out-of-sample utility pilot; confirmatory U2
requires at least 59 eligible clusters screened in public-catalog order, or is
reported as underpowered if the catalog is exhausted.

The first smoke attempt exposed two reproducibility hazards and produced no
update. It used LeRobot's incomplete default cache and received an anonymous
Hugging Face HTTP 429 while fetching missing shards. Both training launchers now
require an explicit LeRobot cache root and reject caches that do not contain the
frozen 1,693-file dataset. A second launch correctly refused to overwrite the
empty checkpoint directory created by that failed attempt, so the successful
run used a fresh experiment name and retained the failed directory for audit.

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

The final `libero_goal` and `libero_10` runs completed under the same frozen
protocol. Goal succeeded in 490/500 episodes (98.0%), exactly matching the
pinned public result. LIBERO-10 succeeded in 465/500 episodes (93.0%; Wilson 95%
interval 90.4--94.9%), compared with the pinned 92.4%. A strict combiner
accepted the four suites only after verifying 500 episodes in each and produced
1,939/2,000 = 96.95% overall (micro Wilson 95% interval 96.1--97.6%). The EGL
destructor warning appeared only after the evaluator had written all 500 final
records and does not change the accepted outcome table.

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

A direct paired scene-cluster bootstrap of commitment minus formation timing is
positive in all 15 eligible states for every retained property. The mean gap is
9.53 flow steps for the full action and translation (95% interval 9.27--9.80),
8.80 for rotation (8.60--9.00), and 8.20 for target direction (6.80--9.40).
The corresponding median gaps are 10 (interval 9--10), 9 (9--9), and 10
(9--10). Thus the formation--commitment separation is a directly estimated
within-state effect rather than an inference from nonoverlapping summaries.

The complete 15-state coarse layer and action-dimension grid adds population-
level causal localization. Scene-cluster bootstrap intervals, exact sign-flip
tests over the 15 state clusters, and Benjamini--Hochberg correction within each
property--intervention family all use the frozen raw grid. For full action,
translation, and rotation, the largest residual-stream transfer occurs at flow
step 9 and final action-expert layer 17: mean symmetric NCTE 0.0955 (95% interval
0.0921--0.0990, q=`6.95e-5`), 0.0947 (0.0908--0.0987, q=`7.09e-5`), and 0.0990
(0.0955--0.1020, q=`7.09e-5`). There are 43, 42, and 63 FDR-significant positive
cells respectively out of 180 tested sites.

Property-matched `x_t` interchange at the final flow step is substantially
stronger. Translation-coordinate transfer explains mean symmetric translation
NCTE 0.537 (0.517--0.559, q=`1.83e-4`), while rotation-coordinate transfer
explains rotation NCTE 0.900 (0.873--0.922, q=`1.26e-4`). The corresponding
full-action peak is the translation-coordinate `x_t` interchange at step 9,
NCTE 0.468 (0.439--0.498, q=`1.26e-4`). These effects establish late geometric
instantiation at the action state and a smaller but reproducible final-layer
mediator; they do not imply that earlier significant sites are unnecessary or
that the residual patches form a complete causal decomposition.

A clean-only pre-contact pilot tested whether gripper closure could be brought
inside the 10-token horizon without changing the prompt-only target-pair design.
States were forked one full horizon before each origin rollout's first contact.
In the base-derived state, the wine prompt first closed at token 9 while the bowl
prompt did not close in the initial chunk, but the bowl rollout contacted the
wine bottle before its instructed bowl. In the donor-derived state, both prompts
closed at token 8; the wine rollout contacted the bowl first and failed. Both
blocks therefore fail preregistered property-specific eligibility for different
reasons. They are retained as a negative identifiability result: bringing
closure into the horizon also places the robot close enough to the origin object
to invalidate the counterfactual instruction. Gripper timing is not inferred
from these data and requires a separately registered target-pose family.

The repeated-intervention closed-loop target sweep completed on all 15
preregistered dual-success scene states, yielding 30 directional seed-0 curves.
All 30 source-retention curves are monotonic, all full-donor endpoints transfer
target identity, and all full-source endpoints retain it. Source retention is
exactly zero through boundary 7, rises to 0.600 at boundary 8 (scene-cluster
bootstrap 95% interval 0.433--0.767), and is exactly one at boundaries 9 and 10.
Eighteen directions commit at boundary 8 and 12 at boundary 9; the median is 8,
interquartile range 8--9, and source-retention AUC 0.210. One boundary-1 rollout
selected neither registered target, so donor transfer is 0.967 there while
source retention remains zero; the raw categorical outcome is retained. This is
the final clean-selected seed-0 aggregate, not a partially completed interim.

## 2026-08-31: phase-aligned destination positive control

Four exact prompt-only destination families were generated from the official
`libero_goal` registry. The first phase-aligned pilot uses the bowl-on-stove
versus bowl-on-plate family. One clean rollout was captured for each prompt, and
the earliest state satisfying the frozen post-grasp rule was selected: five
consecutive control steps with gripper--bowl contact and at least 2 cm of lift.
The base-derived snapshot was step 51 (2.32 cm lift, persistence through step
55); the donor-derived snapshot was step 50 (2.60 cm lift, persistence through
step 54). An initial implementation incorrectly read LIBERO's static placement
specification as the live bowl pose and reported zero lift. The audited fix uses
the mean live MuJoCo position of the bowl's contact geoms; the selection
threshold and persistence rule were unchanged.

Both prompts succeeded from both serialized held-bowl snapshots. A registered
endpoint evaluator based on final live bowl position selected the instructed
destination in all four clean rollouts, with nearest-destination margins of
0.330--0.378 m. From the base-derived snapshot, the repeated full-donor flow
condition then swapped the physical destination in both directions: the stove
prompt placed nearest the donor plate (0.378 m margin), and the plate prompt
placed nearest the donor stove region (0.342 m margin). Images, wrist images,
proprioception, and simulator state remained exact. LIBERO's built-in success
flags were false for these crossed rollouts because each environment still
checks its source BDDL goal; they are not used as the donor-destination endpoint.
This establishes a causal subgoal-transfer ceiling after target identity and
grasp are already fixed. The boundary curve and independent donor-derived state
replication remain in progress.

For runtime only, the full destination curve stops after five consecutive steps
with the released bowl within 8 cm of either registered destination. The radius
was calibrated from clean endpoint distances (2.7 and 5.1 cm) and does not alter
the nearest-destination scorer. The shortened full-donor boundary must reproduce
the original 400-step endpoint before the rule is used for the remaining curve.

The shortened boundary reproduced both original donor destinations to within
`3.7e-5` m of the long-run margins and reduced the two rollout lengths from 400
steps each to 64 and 83. The complete base-derived destination curve then passed
both endpoint controls and was monotonic in both directions. Boundaries 0--6
selected the donor destination in both directions, boundary 7 split by
direction, and boundaries 8--10 selected the source destination in both
directions. Direction-specific categorical commitment boundaries were 7 and 8,
with source-retention AUC 0.300. This is the first full causal subgoal-timing
curve; the donor-derived physical-state block is the preregistered replication.

The donor-derived block subsequently completed with the same result. Both
directional curves were monotonic, both endpoint controls passed, the two
direction-specific commitment boundaries were again 7 and 8, and AUC was
0.300. Combining the two independently selected physical-state blocks yields
four monotonic directional curves, 100% full-donor transfer, 100% full-source
retention, a median commitment boundary of 7.5 (interquartile range 7--8), and
the same AUC of 0.300. Source retention is exactly zero through boundary 6,
0.5 at boundary 7, and one at boundaries 8--10 in both blocks. The matching
step function across opposite-origin snapshots is a replicated causal result,
although the current inference remains conditional on one destination contrast
and one shared-noise seed.

For the gripper-timing follow-up, three target-pose fixtures were generated from
the base-derived pre-contact snapshot at step 18 by translating only the wine
bottle 2, 4, or 6 cm along the planar end-effector-to-target axis. Instruction
and robot state are identical within each pair. A simulator-level audit requires
bitwise equality of every generalized position outside the target free joint's
two planar coordinates, as well as every velocity and actuator state. Clean
closed-loop eligibility screening is performed without viewing any patched
outcome. The primary selection rule was frozen as the smallest offset passing
all clean criteria. The 2 cm candidate passes: both initial inputs and restored
simulator states are exact, both sides contact only the wine bottle first, and
both succeed. The unshifted pose first closes at action-token position 9,
whereas closure is right-censored after token 9 for the shifted pose. The 2 cm
candidate was therefore frozen for causal intervention before inspecting the
larger-offset screens or any patched outcome. The subsequently completed clean
sensitivity grid found that all three offsets remained exact, first-target
contact-valid, and dual-successful. The 4 cm candidate closed at token 9 on both
sides and is closure-ineligible; the 6 cm candidate reproduced the 2 cm
finite-versus-censored contrast. This nonmonotonic clean policy response does not
alter the already frozen smallest-valid-offset choice. Both frozen 2 cm online
first chunks are bitwise identical to the independent offline intervention
runner's clean endpoints (maximum absolute error zero).

The frozen 2 cm causal coarse sweep passed all controls exactly: 60 grouped-
dimension identity interventions, nine residual identity interventions, and the
full-donor switch had maximum absolute action error zero. Encoding absent
closure as right-censored at the 10-token horizon, the clean closure-time
contrast is token 9 versus 10. It is absent from both intermediate clean
estimates through flow step 6 and appears exactly at step 7, remaining stable
thereafter. The symmetric suffix-switch curve is donor-like through boundary 3,
directionally split at boundaries 4--6, and source-retaining in both directions
from boundary 7, giving closure formation and commitment steps of 7. Swapping
only the gripper coordinates of `x_t` at flow step 6 transfers the categorical
closure time completely in both directions; translation- or rotation-coordinate
swaps never do so. In the same pair, continuous translation and rotation do not
reach 0.8 retention until boundary 10, so this clean pilot falsifies a universal
"gripper always commits last" hierarchy.

A labeled post-hoc threshold sensitivity leaves the clean finite-versus-censored
endpoint unchanged at gripper-command thresholds -0.5, 0, and 0.5. Commitment
is boundary 6 at -0.5 and boundary 7 at 0 and 0.5; formation is steps 6, 7, and
8 respectively. The qualitative mid-to-late-flow result is therefore stable,
while the exact discrete boundary has the expected one-step threshold
sensitivity.

Before single-token outcomes are inspected, the follow-up grid is frozen at
flow steps 6--8 and layers 0, 8, 14, and 17. Step 6 is the first bidirectionally
effective gripper-`x_t` site, layer 14 is the earliest residual layer that changes
closure there, and layers 0, 8, and 17 are depth anchors. All ten future action
positions are tested at every selected site; the position set is not reduced
after viewing token effects.

The frozen single-token grid completed with 120 bidirectional layer--flow--token
sites and exact identity/full-donor controls. Exactly nine sites changed the
categorical closure time, and all nine patched future action position 9. Every
one of the 111 patches to positions 0--8 had zero closure-time effect. At flow
step 6, a token-9 effect appeared only at the final tested layer 17; at steps 7
and 8, token 9 was effective at all four depth anchors. Each nonzero site was
direction-asymmetric (symmetric NCTE 0.5), so the crisp positional localization
is stronger than the current evidence for a layer-specific mediator. This is a
pilot result and requires scene-state and noise replication.

The preregistered four-noise sensitivity then found that all eight clean
rollouts remained exact, first-target-contact-valid, and successful, but closure
contrast was noise-conditional. Only seed 0 produced token 9 versus censored;
seeds 1 and 3 closed at token 9 on both sides, and neither side closed within
the seed-2 chunk. The conservative all-seed gate therefore marks zero eligible
state clusters for gripper closure, even though one of four seed units is
eligible. The seed-0 causal and token-localization results remain valid for that
sampled action mode but are not a noise-robust state-level effect. By contrast,
translation, rotation, target direction, and the full action remain eligible in
all four seeds; their four-seed commitment boundaries are 10, 10, 9, and 10.

The first closed-loop recovery pilot used the instruction pair alphabet soup
versus cream cheese and replaced only the first receding-horizon chunk. Boundary
10 reproduced both clean full-rollout outcomes exactly; boundary 0 reproduced
the opposite side's first action chunk bitwise. All initial model inputs and
restored simulator states were exact, the intervention was applied only at
replan zero, and all four endpoint rollouts succeeded. These controls do not,
however, establish recovery. In the donor-side identity rollout the policy had
already contacted alphabet soup before cream cheese, so the corresponding
patched wrong-object contact was not induced by the donor chunk. The corrected
eligibility rule requires both source-first contact under the identity control
and donor-first contact under the patched chunk. Zero of two directions pass
that rule; recovery rates are therefore undefined and interpretation is
disabled. The pilot is retained as a negative result and as a regression test
against conflating eventual success after a patch with causally induced
recovery.
