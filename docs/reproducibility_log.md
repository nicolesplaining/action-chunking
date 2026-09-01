# Reproducibility log

## 2026-08-31: official pi0.5-LIBERO baseline replication

The unchanged official OpenPI evaluator completed 500 episodes per suite for
three suites. Spatial achieved 493/500 = 98.6% (Wilson 95% interval
0.971--0.993), object 491/500 = 98.2% (0.966--0.991), and goal 490/500 = 98.0%
(0.964--0.989). The public pinned OpenPI table reports 98.8%, 98.2%, and 98.0%
respectively, so object and goal match exactly and spatial differs by 0.2
percentage points. LIBERO-10 remains in progress. Task-level tables and raw
logs are retained; the four-suite average will not be reported until all 2,000
episodes complete.

The matched public pi0 base checkpoint was converted with the pinned OpenPI
converter. Because the public pi0 checkpoint does not provide LIBERO
normalization statistics, the unchanged public `pi0_libero` data configuration
was run over all 8,545 batches of `physical-intelligence/libero`. The resulting
`norm_stats.json` is 1,951 bytes with SHA-256
`f68a5fafe15e1577b7bb2c6fc4837a7d1669e2e9be3752f2589c3d327c6f8ccf`.
The access credential was supplied only to the downloader process and was not
written to the repository, shell history, or experiment metadata. These frozen
statistics are the input to the preregistered two-device pi0 control training.

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

The first eight completed scene states in the repeated-intervention closed-loop
target sweep provide 16 directional seed-0 curves. All 16 are monotonic, all
full-donor endpoints transfer target identity, and all full-source endpoints
retain it. Source-target retention is zero through boundary 7, rises to 0.5625
(cluster-bootstrap 95% interval 0.375--0.750) at boundary 8, and is 1.0 at
boundaries 9 and 10. The median directional categorical commitment boundary is
8, with interquartile range 8--9 and source-retention AUC 0.206. This is a
completed-job-only interim; the final aggregate requires all 15 preregistered
states and does not read the partially written current job.

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
