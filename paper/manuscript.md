# When can a VLA action sample still be retargeted? Causal editability of flow-matching action chunks

## Abstract

Action-chunking vision-language-action (VLA) policies predict multiple future
controls in one inference call, but it is unclear when semantic and geometric
properties of the chunk remain conditionally editable. We study the public pi0.5-LIBERO
policy using paired episodes that share images, robot and simulator state, and
Gaussian action noise while differing in one registered variable. We distinguish
*formation*, when a clean intermediate action estimate resembles the final
paired contrast, from *conditional editability*, measured by whether switching
the conditioning for the remaining flow updates redirects the output. Across 15 dual-success
scene states for a wine-bottle-versus-bowl instruction contrast, full-action and
translation contrasts satisfy the frozen formation criterion at the first flow
estimate, yet lose conditional editability only at the final update in the
aggregate. The within-state formation-to-boundary gap is positive in all 15
states (mean 9.53 of 10 flow
updates for both full action and translation; scene-cluster 95% intervals
9.27--9.80). Repeated closed-loop interventions yield 30/30 monotonic target
curves: behavioral target identity remains fully donor-controlled through seven
updates, is source-retained with probability 0.60 after eight, and is fully
source-retained after nine. Residual-stream interchange peaks at the final
action-expert layer and final flow step (normalized causal transfer 0.095--0.099
across full action, translation, and rotation), while property-matched final-step
action-state swaps transfer 0.537 of translation and 0.900 of rotation. A
post-grasp destination contrast replicates across two independently selected
physical-state blocks, losing editability at boundaries seven or eight. A closure-time
case localizes completely to future action position nine, but only one of four
noise modes is eligible and therefore supports no population claim. These
results show that early appearance does not imply loss of counterfactual
control: properties can be legible throughout integration while remaining
causally editable until late updates. They do not show that the model was
undecided or lacked an internal plan. A registered inference experiment tests
whether the measured boundary enables successful retargeting without restarting
generation, while a matched pi0 control is trained before cross-model claims.
If its frozen competence and conversion-parity gates pass, the control is run on
the clean-eligible scene intersection with the same ten flow updates and layer
sites. Direct position contrasts use indices zero through nine, while a
ten-bin normalized-chunk-time sensitivity retains pi0's full 50-action horizon.

## 1. Introduction

Action chunking reduces closed-loop inference frequency by predicting a sequence
of future controls rather than one action at a time [@zhao2023act]. Flow- and
diffusion-based policies add another temporal axis: the entire action sequence
is iteratively constructed from noise [@chi2023diffusionpolicy;
@black2024pi0]. A resulting action can therefore depend on at least three
computational coordinates: transformer depth, future position within the chunk,
and flow-integration time. Existing VLA work demonstrates that internal features
can be interpreted or causally steered [@haon2025mechanistic;
@swann2026drvla; @jin2026eventsae], but does not establish when natural
counterfactual conditioning ceases to redirect an action sample.

We ask: **when can each property of a flow-matching VLA action sample still be
retargeted?** A clean intermediate prediction may already point toward the final
object or trajectory, yet later conditioning can still redirect it. We therefore
separate descriptive formation from conditional editability. Formation measures the
agreement of the intermediate clean-action contrast with the final contrast.
Editability uses a suffix conditioning switch: early flow updates use one member
of a minimally different episode pair and all remaining updates use the other.
This intervention tests whether natural paired conditioning still has causal
control over the final property. Successful redirection is compatible with an
already formed but overwritable plan and can reflect strong instruction
following; it is not evidence of indecision or representational absence.

Our initial study uses the public pi0.5-LIBERO checkpoint
[@physicalintelligence2025pi05] and the public LIBERO simulator and task
definitions [@liu2023libero]. The design follows causal interchange principles
[@geiger2022iit; @geiger2023causalabstraction] and activation-patching best
practices [@zhang2023patching]: both directions are measured, identity and
full-donor controls are exact, normalized effects are withheld when clean
endpoints do not differ, uncertainty is clustered by physical scene state, and
all heatmap families receive Benjamini--Hochberg correction.

The principal mechanistic finding is a temporal dissociation. Geometric contrasts are
present in the first clean estimate but remain causally editable through most
or all of the ten-step flow trajectory. Behavioral target identity and a
post-grasp destination become fixed only during the last two or three updates.
Late action-state coordinates carry large property-matched effects, whereas
single residual-layer patches are smaller and include both positive and
counter-directed transfer. A future-token closure result is sharply localized
but noise conditional, illustrating why mechanistic case studies and
population conclusions must remain separate. The consequential question is
therefore evaluated separately: whether this boundary predicts the last point
at which a newly instructed task can be reached by continuing the current
sample with fewer post-update flow evaluations than a full restart.

## 2. Related work

Pi0 introduced a flow-matching VLA with a vision-language prefix and action
expert that generates continuous action chunks [@black2024pi0]. Pi0.5 extends
the system with heterogeneous co-training and open-world generalization
[@physicalintelligence2025pi05]. We use Physical Intelligence's public OpenPI
implementation at commit
`215abfb217dbac7d5f1273282331b9b1866c0479`; executable checkpoint behavior,
rather than unexposed training components, defines the object of study.

ACT established the effectiveness of transformer action queries for chunked
control [@zhao2023act], while Diffusion Policy demonstrated iterative denoising
of action sequences with receding-horizon execution [@chi2023diffusionpolicy].
Our question is complementary: rather than comparing policy classes, we resolve
where and when a fixed trained model causally constructs a chunk.

Iterative action generation also creates a latency problem. One-Step Diffusion
Policy distills a diffusion controller and reports both task success and action
frequency [@wang2025onedp], while One-Step Flow Policy applies self-distillation
to flow policies and includes a pi0.5 integration [@li2026ofp]. Our practical
experiment does not distill or retrain the model. It addresses a different
online event: when an instruction changes after some flow evaluations are
already sunk, can the current action state be reused rather than restarted?
SafeDiffuser embeds constraints into iterative diffusion planning and separates
safety satisfaction from planning quality and overhead
[@xiao2025safediffuser]. We adopt that measurement separation for the planned
obstacle study without claiming formal barrier guarantees.

Interchange interventions replace an internal variable in one computation with
its value from a counterfactual computation, providing a test of causal
abstraction rather than correlation [@geiger2022iit;
@geiger2023causalabstraction]. Activation-patching conclusions can depend
strongly on the metric and corruption or counterfactual choice
[@zhang2023patching]. We avoid arbitrary corruption by using natural paired
episodes, report directional asymmetry, and distinguish full-donor ceilings from
single-site effects.

Recent VLA interpretability studies causally steer FFN vectors
[@haon2025mechanistic], identify sparse features in pi0.5 and related models
[@swann2026drvla], and ground sparse features in rollout events
[@jin2026eventsae]. Those results motivate causal VLA analysis and cross-scene
replication. Our contribution is an intervention grid crossing every flow step,
action-expert depth, action-state group, and—within an eligible case—future
action position, together with a separate behavioral editability estimand.

## 3. Methods

### 3.1 Model, benchmark, and public-source reuse

The primary policy is the public `pi05_libero` checkpoint executed with the
pinned OpenPI source. The resolved model has a ten-action horizon, 32 padded
action dimensions, a Gemma 2B vision-language configuration, and an 18-layer
Gemma 300M action expert; LIBERO consumes the first seven physical action
dimensions. We use the pinned public LIBERO task registry and BDDL definitions,
not task semantics reconstructed from filenames. Model loading, preprocessing,
normalization, flow integration, and benchmark evaluation reuse the public
OpenPI code. Local code only adds checked intervention hooks, paired-fixture
generation, deterministic evaluators, and analysis.

The official evaluator runs 50 trials for each of ten tasks in every LIBERO
suite. Completed logs are accepted only if they contain exactly ten task names,
50 completed trials per task, and 500 episodes per suite. The reproduced counts
are 493/500 for Spatial (98.6%), 491/500 for Object (98.2%), 490/500 for Goal
(98.0%), and 465/500 for LIBERO-10 (93.0%). Thus the strict four-suite aggregate
is 1,939/2,000 (96.95%; micro Wilson 95% interval 96.1--97.6%), compared with
96.85% in the pinned OpenPI table. Partial suite results are not used as a
competence claim.

### 3.2 Paired episodes and eligibility

Each pair shares the serialized simulator state, main and wrist images, robot
state, preprocessing, and saved Gaussian action noise. Exactly one named
variable differs. The primary instruction pair changes the instructed object
between a wine bottle and a black bowl in the same LIBERO Goal scene. Sixteen
scene states were generated and cleanly replayed; 15 satisfy the registered
dual-success criterion and define the current state-level analysis. Selection
uses only clean endpoints, never patched outcomes.

For destination timing, we compare placing a held bowl on the stove versus on a
plate. The phase anchor is the earliest state beginning five consecutive control
steps with gripper--bowl contact and at least 2 cm live MuJoCo lift. We retain
separate base-derived and donor-derived physical-state blocks. Both prompts must
succeed from both restored blocks, and clean destination margins must pass before
intermediate flow boundaries are interpreted.

For closure timing, only the target's two planar free-joint coordinates move.
The registered 2 cm shift is along the end-effector-to-target axis; all other
generalized positions, velocities, actuator state, instruction, and robot state
are exact. Eligibility requires dual success, instructed-target-first contact,
and a clean within-chunk closure-time contrast. Closure absence is right-censored
at the ten-token horizon. Noise seeds are repeated modes, not independent scene
states; a state-level effect requires eligibility in every registered seed.

### 3.3 Formation

At flow time `t`, with noisy action state `x_t` and predicted velocity `v_t`, the
public linear flow parameterization gives the intermediate clean estimate

```text
a_hat(t) = x_t - t v_t.
```

For each property, the paired intermediate contrast is compared with the final
clean contrast. Formation is the earliest step whose relative contrast error is
at most 0.20 and remains within tolerance at every later step. Formation is a
descriptive property of the clean computation and is never labeled causal.

### 3.4 Conditional editability by suffix conditioning switch

For a pair `A, B`, a switch at boundary `k` integrates the same initial noise
under condition `A` for the first `k` Euler updates and condition `B` for all
remaining updates. For a continuous scalar or vector property `Y`, oriented
source retention is

```text
R(k) = 1 - <Y_switch(k) - Y_A, Y_B - Y_A> / ||Y_B - Y_A||^2.
```

The reported curve averages the two directions after orientation. The
editability boundary is the earliest boundary at which isotonic source
retention is at least 0.80 and remains so thereafter. We also report retention
AUC; `0.5 - AUC` measures late weighting relative to a uniform-update reference
but does not imply a discrete planning stage.

Behavioral target identity is evaluated online. The same boundary intervention
is applied at every replan until first task-object contact, producing source,
donor, or neither-target outcomes. Destination identity is evaluated from the
phase-aligned held-object state using the final live object position relative to
registered endpoints. LIBERO's source-task success flag is not used to label a
crossed donor destination.

### 3.5 Retargeting without restarting generation

The practical experiment simulates a new instruction arriving after `k` flow
evaluations under the old instruction. A *continue* strategy retains the live
action state `x_k`, prepares the new image--language condition, and evaluates
only the remaining velocity fields. A *restart* strategy discards `x_k` and
regenerates from the same original Gaussian noise under the new instruction.
The new instruction defines the evaluated LIBERO task and every subsequent
clean replan. Outcomes are new-target-first contact, eventual task success,
completion steps, synchronized post-event latency including condition
preparation, and post-event velocity evaluations.

This construct is deliberately independent of whether the hybrid action is
close to a donor vector. The existing target curve freezes boundaries
`k in {0, 7, 8, 9, 10}` before utility outcomes: 0 is the full-restart ceiling,
7 is the predicted safe continuation point, 8 is the transition, and 9--10 are
negative controls. At `k=7`, continue executes three post-event velocity
evaluations versus ten for restart. A practical claim requires held-out scene
states, byte-exact `k=0` equivalence, a five-percentage-point paired
noninferiority margin for both target-first contact and task success, and lower
measured post-event latency.

### 3.6 Layer, action-state, and token interventions

At flow step `s` and action-expert layer `l`, residual interchange replaces all
action-token post-layer residuals in the recipient with their donor values. At
selected sites, future positions are patched individually. Action dimensions
are intervened on in the evolving action state `x_t` or velocity `v_t`, grouped
as translation, rotation, or gripper; action dimensions are not treated as
hidden residual channels.

For continuous outcomes, normalized causal transfer is

```text
NCTE = <Y_patch - Y_A, Y_B - Y_A> / ||Y_B - Y_A||^2,
```

where zero is the recipient endpoint and one is full donor transfer. Effects are
not clipped. Every patch is run in both directions. Identity patches must match
the clean action exactly, and full-donor interventions define the transfer
ceiling.

### 3.7 Statistics

The physical scene-state hash is the resampling cluster. Primary intervals use
10,000 scene-cluster bootstrap replicates with frozen seeds. Cellwise two-sided
sign-flip tests are exact when feasible, and Benjamini--Hochberg correction is
applied within property and intervention family. A positive mediation marker in
the manuscript requires both adjusted `q < 0.05` and a cluster 95% interval
strictly above zero. Effective state counts accompany all tables. With one
eligible scene state, intervals and `q` values cannot support population
inference; such results are labeled descriptive.

## 4. Results

### 4.1 Clean formation precedes loss of conditional editability

All 15 primary scene states are eligible for full action, translation, rotation,
and target direction. Full-action, translation, and rotation contrasts satisfy
the aggregate formation criterion at step 0 with bootstrap intervals 0--0;
target direction also forms at step 0 in the aggregate, with interval 0--3.
Nevertheless, the continuous editability boundary occurs at step 10 for full action,
translation, and target direction, and step 9 for rotation. Retention AUC is
0.232, 0.220, 0.159, and 0.310, respectively.

The state-paired timing gap is positive in all 15 states for all four properties.
Mean editability-boundary-minus-formation gaps are 9.53 updates for full action (95%
interval 9.27--9.80), 9.53 for translation (9.27--9.80), 8.80 for rotation
(8.60--9.00), and 8.20 for target direction (6.80--9.40). Thus early clean
contrast alignment does not mean that the corresponding output is fixed.

### 4.2 Behavioral target identity remains conditionally editable late

The 15-state online experiment yields 30 directional curves, all monotonic. Full
donor controls transfer target identity in every direction and full source
controls retain it in every direction. Source retention is exactly zero through
boundary 7, rises to 0.600 at boundary 8 (scene-cluster interval 0.433--0.767),
and is exactly one at boundaries 9 and 10. Eighteen directional curves cross
the editability threshold at boundary 8 and 12 at boundary 9, giving median 8
and interquartile range 8--9.
One boundary-1 rollout contacts neither registered target; it remains in the raw
table rather than being relabeled. Donor transfer at that boundary is therefore
0.967 while source retention is zero.

### 4.3 Late action-state coordinates carry large property-matched effects

At the final flow step, swapping translation coordinates of `x_t` transfers
0.537 of the translation contrast (cluster interval 0.517--0.559,
`q=1.83e-4`). Swapping rotation coordinates transfers 0.900 of rotation
(0.873--0.922, `q=1.26e-4`). For the full action, the largest grouped effect is
the final-step translation `x_t` patch, NCTE 0.468 (0.439--0.498,
`q=1.26e-4`). Corresponding `v_t` effects are smaller. These interventions
locate property-specific geometry in the evolving action state late in flow
integration.

Residual-stream effects are smaller. For all three reported outcomes, the
largest positive patch is the final action-expert layer (17) at the final flow
step (9): NCTE 0.0955 for full action (0.0921--0.0990), 0.0947 for translation
(0.0908--0.0987), and 0.0990 for rotation (0.0955--0.1020), all with
BH-adjusted `q < 7.1e-5`. The heatmap also contains counter-directed effects in
middle layers at late flow steps. Residual interchanges therefore do not form an
additive decomposition, and layer 17 is not described as a unique symbolic
planning site.

### 4.4 Post-grasp destination timing replicates across physical states

Both independently selected held-bowl state blocks pass all clean endpoint
controls. In all four directional curves, boundaries 0--6 select the donor
destination, boundary 7 splits by direction, and boundaries 8--10 select the
source destination. Each block independently produces editability boundaries 7
and 8; combined median is 7.5 with interquartile range 7--8 and AUC 0.300. The
replication shows that a post-grasp subgoal remains editable until late
integration. Because the target and destination experiments begin at different
rollout phases, their boundary difference is not a universal semantic ordering.

### 4.5 Closure localization is sharp but noise conditional

In the registered seed-0 target-pose case, the unshifted clean chunk first closes
at future action position 9 while the shifted chunk is right-censored at the
ten-position horizon. Closure forms and loses editability at flow step 7. Swapping only
the gripper coordinate of `x_t` at step 6 fully transfers categorical closure
time in both directions; translation and rotation swaps do not.

Across 120 single-position residual sites (three flow steps, four layer anchors,
and ten future positions), exactly nine sites change closure time, all at future
position 9. At step 6, only layer 17 is effective; at steps 7 and 8, position 9
is effective at all four anchors. However, only seed 0 of four has an eligible
clean closure contrast. Seeds 1 and 3 close at position 9 on both sides, while
neither side closes within the seed-2 chunk. The conservative all-seed gate
therefore yields zero eligible scene clusters. This is a mechanistic case study
for one sampled action mode, not a state-level population result.

### 4.6 Matched pi0 control status

The public pi0 base checkpoint and upstream `pi0_libero` full-finetuning recipe
are the registered model-level control. Dataset normalization traversed all
8,545 public batches; the frozen statistics file has SHA-256
`f68a5fafe15e1577b7bb2c6fc4837a7d1669e2e9be3752f2589c3d327c6f8ccf`.
A two-update, two-H100 FSDP smoke completed with finite first-reported loss
0.1812, gradient norm 3.7253, parameter norm 1377.8652, and a finalized Orbax
checkpoint. The 30,000-update training and competence evaluation must complete
before any pi0-versus-pi0.5 timing contrast is inserted here.

### 4.7 Registered inference-utility experiment

Dynamic-retargeting outcomes are not yet reported. The continuation and restart
paths, compute accounting, latency instrumentation, eligibility gates, tested
boundaries, and noninferiority margin were frozen after the construct-validity
critique but before running held-out retargeting episodes. This section will
report negative as well as positive outcomes, including the possibility that
the editability boundary predicts immediate target correction but not eventual
success after subsequent clean replanning.

## 5. Discussion

The results support a distinction between *representation* and *control*. The
paired action contrast is already close to its final form in the first clean
estimate, yet replacing the remaining conditioning can redirect the output for
most of integration. This establishes retained counterfactual control, not
indecision: an existing plan may remain overwritable, and late redirection may
be a desirable instruction-following capability.

The largest causal effects occur in late action-state coordinates, consistent
with iterative geometric instantiation. Residual-stream patches identify a
reproducible late mediator but transfer only about one tenth of the paired
contrast at their peak. The mixture of positive and counter-directed residual
effects suggests distributed computation and compensatory dynamics rather than
a single layer that stores a complete plan.

Behavioral target and destination curves both transition late but are not
directly exchangeable: destination is measured after grasp from a held-object
state, whereas target identity begins before contact. A stronger hierarchy claim
would require the same property family measured at matched rollout phases.

The closure case demonstrates the value and danger of token-localized analysis.
Its position specificity is exact for the sampled mode, but the clean contrast
itself is not stable across noise. Treating four noise modes as four independent
states would incorrectly turn a conditional event into a population claim. Our
all-seed gate prevents that error.

## 6. Limitations and pending registered experiments

The current 15-state target analysis uses one shared noise seed for its
population grid. It supports scene-state replication but not a complete
state-by-noise factorial estimate. The destination replication uses two physical
state blocks, one task contrast, and one noise seed. Gripper closure has no
noise-robust eligible state and remains descriptive.

Rotation of the action command is not yet equivalent to a grasp-frame outcome;
a phase-aligned grasp-orientation evaluator remains required before claiming a
grasp-orientation editability boundary. The registered obstacle-position and closed-loop
recovery families are also pending. In an initial first-replan recovery pilot,
all exactness controls passed and all patched rollouts eventually succeeded, but
zero of two directions had a donor-first contact that was induced relative to
its identity control. Recovery rates are therefore undefined for that pair.
This negative result shows why recovery must distinguish an
intervention-induced failure event, the first post-perturbation chunk, and
eventual task success; it cannot be inferred from success after a patch alone.

The pi0 comparison is a matched protocol and dataset control, not a pure
architectural intervention: pi0 and pi0.5 differ in pretraining, checkpoint
history, action horizon, and released recipe. Pi0.7 is excluded because the
pinned public OpenPI release contains neither internal weights nor an
intervention-compatible implementation. A closed endpoint would support only a
separately labeled behavioral comparison.

Interchange patches create hybrid computations. Exact identity controls and
natural paired donors reduce but do not eliminate distribution-shift concerns.
NCTE is a directed projection, not a claim that patched sites compose additively
or encode human-like symbolic plans.

## 7. Reproducibility and release plan

The repository pins OpenPI and LIBERO revisions and versions every pair
generator, intervention, evaluator, statistical test, and figure builder. Raw
artifacts retain paired manifests, simulator-state hashes, saved noise,
bidirectional outcomes, and exact controls. Manuscript figures are generated
from immutable CSV tables; a figure manifest records every input SHA-256.
Machine-specific hosts, credentials, and checkpoint paths are not versioned.

The final release will include accepted and excluded pair manifests, all raw
clean and patched outcome tables, environment locks, checkpoint provenance,
confidence intervals, FDR tables, negative results, and the matched pi0
artifacts after their registered competence gate passes.
