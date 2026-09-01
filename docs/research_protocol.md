# Research protocol: causal editability and retargeting in action chunks

Protocol version: 0.22 (matched-control common-state gate)

## 1. Research question

When can the semantic and geometric properties of a flow-matching action sample
still be redirected, which internal sites mediate that redirection, and does the
measured boundary predict useful low-latency correction?

The primary model is the public pi0.5-LIBERO checkpoint. A pi0 model fine-tuned
with the pinned public `pi0_libero` recipe uses the same demonstrations, 30,000-
step budget, and evaluation scenes as the matched model-level control. Its
architecture-specific 50-action horizon and legacy extra-delta transform differ
from pi0.5's 10-action horizon and are reported rather than disguised as perfect
matching; primary cross-model position comparisons use the first 10 positions
and a normalized chunk-time sensitivity analysis. Pi0.7 is outside the
confirmatory scope until its weights and internal implementation are public.

The pi0 control is trained with the pinned public JAX `pi0_libero` recipe for
30,000 optimizer steps, global batch size 32, two-device FSDP, and its configured
0.99 EMA; normalization statistics are computed with OpenPI's unchanged public
script. This preserves the released model-specific recipe and EMA rather than
introducing a second training implementation. It does not equalize total sample
exposures to the released pi0.5 checkpoint, so the comparison is explicitly a
model-level control, not an isolated causal estimate of one architectural flag.

The computational axes are:

- flow integration step `s`;
- action-expert transformer layer `l`;
- future action position `p`;
- robot action dimension or preregistered dimension group `d`.

The population action-token follow-up uses all 15 clean-eligible target states,
flow steps `{0, 7, 8, 9}`, action-expert layers `{0, 8, 14, 17}`, and each of
the ten pi0.5 action-token positions. This 160-cell grid was frozen before any
population target-token outcome was generated. At flow step 9 and layer 17,
the primary token-position contrast is the scene-cluster mean NCTE over the
five positions executed before replanning minus the mean over positions 5--9.
The contrast is tested separately for each preregistered eligible property with
a scene bootstrap interval and exact sign-flip test. The complete grid remains
visible and uses BH adjustment within each property family; the primary grouped
contrast is not selected from the heatmap peak.

The behavioral properties are target identity, initial trajectory direction,
grasp orientation, gripper-closure timing, and post-perturbation recovery.

## 2. Conceptual distinctions

### 2.1 Formation

At flow time `t`, pi0.5 predicts a velocity `v_t` for the evolving noisy action
tensor `x_t`. Under the public model's linear flow parameterization, the
intermediate clean-action estimate is

```text
a_hat(t) = x_t - t * v_t.
```

A property has *formed* when an evaluator applied to `a_hat(t)` reliably agrees
with the final unpatched action and remains stable at later steps. Formation is
descriptive, not by itself causal.

### 2.2 Layerwise causal mediation

For a base episode `A` and a minimally different donor episode `B`, an
interchange intervention replaces an action-token residual activation in the
base run with the corresponding donor activation:

```text
h_A[s, l, p] <- h_B[s, l, p].
```

The remaining layers and flow steps run normally. This measures whether the
site mediates the paired behavioral contrast. Because action-expert hidden
states are recomputed at every flow step, a transformer layer is not a
persistent memory location across the entire trajectory.

### 2.3 Flow-step conditional editability

Conditional editability is measured with a suffix conditioning switch. The action state is
integrated using condition `A` for the first `k` Euler updates and condition `B`
for all remaining updates. Let `Y` be a continuous property score oriented from
`A` to `B`. Retention is

```text
R_z(k) = 1 - (Y_switch(k) - Y_A) / (Y_B - Y_A).
```

The symmetric estimate averages `A -> B` and `B -> A` after orienting both to
retention of the initial condition. The editability boundary is the earliest
`k` whose isotonic mean retention is at least 0.8 and remains at least 0.8 for
every later switch. Confidence intervals are cluster-bootstrapped by scene pair.

Because a threshold crossing alone can obscure whether influence accumulates
uniformly across Euler updates, every curve is also compared with the
`R(k)=k/S` uniform-step null. Report retention AUC, half-editability boundary,
and the marginal retention contributed by the final update. A positive
`0.5 - AUC` is a late-weighting index, not by itself evidence for a discrete
internal planning phase.

This is an operational boundary for causal control by a natural,
in-distribution conditioning counterfactual. Successful redirection does not
show that the model was undecided, had no source-conditioned plan, or lacked a
persistent representation. It can instead show that an existing plan is
overwritable or that instruction following remains effective. Conversely,
failed redirection does not prove absolute irreversibility under arbitrary
perturbations. We therefore reserve *commitment* for shorthand tied explicitly
to this estimand and use *conditional editability* in claims.

The study uses a three-level decision rule. Level 1 is descriptive: a causal
map can establish where natural counterfactual conditioning retains control,
but has no claim to behavioral usefulness. Level 2 is predictive: a
scene-level editability boundary must predict the last successful no-restart
correction on sealed outcomes and improve on the frozen fixed-boundary rule.
Level 3 is interventional: a boundary-derived procedure must improve compute
or latency at matched behavior, or improve correction or recovery at matched
cost. Failure at Level 2 or Level 3 leaves Level 1 as a mechanistic result and
precludes claims that the timing is a useful plan-commitment variable.

Instruction following, immediate correction, recovery, persistence, and
eventual task success are separate outcomes. Immediate correction means the
first post-update executed controls follow the new target or constraint.
Recovery additionally requires a registered failure event under the stale
control and its avoidance under the intervention. Persistence asks whether the
effect survives subsequent clean replanning. Eventual success is reported but
cannot by itself establish any of the preceding outcomes.

### 2.4 Registered inference-utility test

The utility experiment is an amendment motivated by construct validity after
the initial editability curves, but before inspecting any dynamic-retargeting
outcome. It simulates a new target instruction arriving after `k` source-
conditioned flow evaluations have already completed. Two inference strategies
share the same observation, Gaussian noise, old instruction, new instruction,
and pre-event action state:

- **continue:** retain `x_k`, prepare the new condition, and execute only flow
  updates `k,...,S-1`;
- **restart:** discard `x_k`, prepare the new condition, and regenerate from the
  original noise with all `S` updates.

The new instruction is used as the task goal and for every subsequent clean
replan. Thus the outcome is not whether a patched action resembles a donor; it
is whether retargeting reaches the newly instructed object, completes that task,
and survives ordinary receding-horizon replanning. The primary tested
boundaries are `k in {0, 7, 8, 9, 10}`: zero is the exact restart ceiling, seven
is the last boundary with complete donor control in the existing population
curve, eight is its transition, and nine and ten are negative controls.

Primary outcomes are new-target-first contact, eventual new-task success,
completion steps, post-event velocity-field evaluations, and synchronized
post-event wall time including new-condition preparation. Continue at `k=7`
uses three post-event velocity evaluations versus ten for restart. It is useful
only if its paired new-target and task-success rates are noninferior to restart
within a frozen five-percentage-point margin and its measured post-event latency
is lower. Scene state is the inference cluster. Full-restart success, exact
initial-state restoration, and byte-exact `k=0` equivalence are eligibility
gates selected without viewing continued outcomes.

The first 16-state out-of-sample block is a utility pilot, not the
noninferiority population. For the frozen five-percentage-point margin, even
zero continuation losses require at least 59 independent eligible scene
clusters for the exact one-sided 95% Clopper--Pearson upper bound on the paired
loss rate to fall below 0.05. This requirement is computed by the versioned
design utility and cannot be met by treating directions, noise seeds, or action
tokens as independent. The confirmatory screen therefore proceeds through
additional exact-scene instruction pairs in the clean public-catalog order
(canonical-scene hash, suite, base task ID, donor task ID, then initialization
index) until at least 59 unique eligible scene-state clusters are frozen. A
cluster is `(suite, canonical-scene hash, initialization index)` and is counted
once even when multiple target contrasts or directions are screened within it.
All screened rows and clusters remain in the denominator. If the public catalog
is exhausted first, U2 remains a pilot estimate and no five-point
noninferiority claim is made.

The immutable expansion artifact combines the pinned `libero_goal` and
`libero_90` strict-pair catalogs. It contains 45 manipulated-object pair
definitions, 2,218 non-pilot screening rows, and 718 unique candidate clusters.
The 32 wine-bottle/bowl initialization states already used for mechanistic or
utility pilots are excluded explicitly. The plan, source catalog hashes,
exclusions, ordering, cluster unit, and 59-cluster stop rule were serialized
before the corrected held-out endpoint screen completed. Eligibility uses only
the frozen old-condition and restart endpoints; no intermediate continuation
outcome can affect where screening stops.

After the endpoint screen reaches its frozen stop rule (or exhausts the public
catalog), an immutable handoff gathers every eligible endpoint row and its
candidate-manifest hash. For the primary utility analysis, exactly one direction
is retained per independent scene cluster: the first endpoint-eligible direction
in the already frozen gate order. Additional eligible directions in that cluster
are recorded but cannot increase the primary denominator. The initial 16-state
pilot and the catalog population use the same rule. All action-only predictions
for a selected population are serialized and hashed before its first continuation
rollout. The catalog screen runs after the pilot regardless of whether the pilot
contains an eligible state; pilot convenience therefore cannot replace the
registered confirmatory population.

For each accepted state, the separately measured first-chunk editability
boundary predicts the last continued boundary that preserves new-target-first
contact and eventual task success. Association is reported with paired
state-level exact accuracy, within-one-boundary accuracy, mean absolute error,
and Spearman correlation. We also report whether the predicted boundary itself
succeeds and whether the immediately following boundary fails. The useful claim requires
out-of-sample scene states and cannot be established from the original 15
mechanistic states alone.

The state-level predictor is frozen before inspecting any intermediate
continuation outcome. On the exact eligible fixture and shared noise, run the
action-only source-to-new-condition flow switch without taking an environment
step, and compute direction-specific target-approach retention over the five
actions that would be executed. The endpoint target-affinity contrast must be at
least 0.01. Let `k*` be the earliest boundary whose isotonic retention reaches
0.8 and remains there; the preregistered predicted last successful continued
boundary is `k* - 1`. Predictions, including invalid contrasts, are serialized
before the closed-loop boundary sweep. The symmetric population curve remains
the primary mechanistic summary, but it is not substituted for this
direction-matched utility predictor.

U1 evaluates all eleven integer boundaries from 0 through 10 so that the last
successful correction is identifiable. U2 remains the primary efficiency test
at boundaries 0 and 7, while boundaries 9 and 10 remain U3's negative controls;
the dense grid is not used to replace those preregistered comparisons.

U1 is useful only if the state-specific action-only predictor adds information
beyond the population rule already available from the mechanistic pilot. Its
frozen comparator always predicts boundary seven, the last boundary with full
donor control in that pilot. Report both predictors' exact rate, within-one
rate, and mean absolute error. The primary comparative statistic is the paired
scene-cluster difference `abs(state prediction - observed) - abs(7 - observed)`.
A predictive positive requires the upper endpoint of its 10,000-resample,
seed-zero scene-cluster bootstrap 95% interval to be below zero. Correlation or
accuracy without improvement over this fixed rule is not called useful timing
prediction.

U2 uses a paired composite endpoint requiring both new-target-first contact and
eventual new-task success. A paired loss is a cluster where restart passes this
composite and continuation at boundary seven does not. The one-sided 95%
Clopper--Pearson upper bound on that loss probability must be strictly below the
frozen 0.05 margin. Directions, target contrasts, tokens, and noise seeds never
enter this denominator. Post-event velocity evaluations must be exactly 10 for
restart and 3 for boundary-seven continuation; otherwise the compute comparison
is invalid. Wall-time savings are summarized by the median paired fractional
reduction and a 10,000-resample scene-cluster bootstrap interval with seed zero.

### 2.5 Post-pilot behavioral-sensitivity amendment

This amendment was added after inspecting the state-zero dynamic-retargeting
pilot and before selecting or evaluating held-out utility states. In that pilot,
boundaries 0, 7, 8, 9, and 10 all reached the new target first and eventually
succeeded. Because boundary 10 includes no new-instruction velocity update, the
initial-state episode was not sensitive to first-chunk editability: five benign
old-conditioned actions were followed by successful clean replanning.

Held-out utility states must therefore pass a failure-induction gate that is
independent of continued boundary outcomes. From the same exact pre-contact
state and shared noise, executing the registered five-action horizon from the
fully old-conditioned chunk must cause the old-target contact or registered
safety event, while a full restart under the new instruction must avoid that
event and preserve clean task competence. States are screened using only these
two endpoint controls. Intermediate continuation boundaries remain sealed until
eligibility is frozen. All screened states and exclusion reasons are retained.

The pre-contact state is the latest clean receding-horizon replan boundary
strictly before first instructed-target contact, rather than an arbitrary
step offset. If that state is at environment step `5j`, the fork uses Gaussian
noise draw `j` from the clean seed-zero sequence. Clean validation saves the
exact external image, wrist image, and proprioceptive array actually supplied
to the policy at every replan. The fork's first policy call uses those saved
arrays rather than a renderer reconstruction from MuJoCo's flat state. Their
hashes must match the registered fixture, and the old-condition endpoint's
first action chunk must be byte-exact to saved clean chunk `j`; otherwise the
candidate is construct-invalid and cannot enter the failure-induction gate.
The physical rollout cannot rely on the saved MuJoCo flat state alone. A
rejected partial screen showed that this restores configuration but not all
transition-relevant robosuite controller and interpolator state. Each candidate
therefore stores every clean executed action from reset through the snapshot
and the corresponding simulator-state prefix. Every fork reproduces the clean
sequence in both task environments: seeded reset, restoration of the registered
initial MuJoCo configuration, then replay of the entire clean action prefix.
The restored initial state and every subsequent simulator state must be
array-exact to the registered prefix, and the final state must be array-exact
to the candidate fixture. Replay arrays, state arrays, model inputs, and the
source chunk are independently hashed. Any mismatch is construct-invalid. This
alignment makes continue, restart, and the clean counterfactual computations
start from the same action-generation event and the same physical transition
state, not merely the same visible configuration.

The primary five-action horizon is unchanged. Executing a longer portion of the
chunk may be reported as a sensitivity analysis but cannot replace the primary
endpoint. U1 and U3 are evaluated only in eligible states; the state-zero pilot
is reported as a negative sensitivity result for those hypotheses and a
positive engineering pilot for U2's compute and latency components.

Every continuation and restart fork must independently pass the same controller-
replay equality gate as endpoint screening; flat-state equality alone is
insufficient. Behavioral outcomes are decomposed before any corrected
continuation is run: old-target contact within the first five executed actions,
new-target contact within those actions, no registered contact before the first
clean replan, first-contact replan index, correct-target/task success rescued
only after clean replanning, and eventual task success. U1's primary composite
remains new-target-first plus eventual task success. The decomposition is
secondary and explains whether failures occur in the intervened chunk, at a
later chunk boundary, or after correct target selection; it does not replace or
redefine the frozen primary outcome.

### 2.6 Registered late visual-safety update pilot

The obstacle-pose pilot is extended to a consequential condition-update test
before inspecting any obstacle intervention outcome. The robot is always
executed in the selected moved-obstacle simulator state. At its first action
generation call, the source condition uses the exactly paired image in which
the distractor remains at its original pose, while the donor condition uses the
live moved-obstacle image; prompt, robot proprioception, initial Gaussian noise,
and every non-obstacle simulator coordinate are held fixed. All later replans
use the live moved-obstacle observation without intervention.

Run full restart and continue-from-`x_k` at every integer boundary `k=0..10`.
The pilot is behaviorally eligible only if the fully source-conditioned first
chunk (`k=10`) contacts the registered obstacle within the first five executed
controls, while full restart under the moved-obstacle condition avoids that
contact and eventually completes the unchanged task. Boundary-zero continue
and restart actions must be exactly equal. Every rollout must exactly restore
the registered simulator state and robot proprioception; the counterfactual
source image and live donor image must each exactly equal their frozen fixture.
Only restart and `k=10` are run for endpoint screening. Boundaries `0..9`
remain sealed and are evaluated only after that endpoint gate passes.

A no-restart safety correction is counted only when it avoids obstacle contact
through the first five controls and eventually completes the task. Report the
entire collision, task-success, minimum-clearance, and latency curves rather
than selecting a favorable boundary. The last successful continued boundary is
descriptive in this one-state pilot. A practical positive requires at least one
`k>0` continuation to meet the composite endpoint while using exactly `10-k`
post-update velocity evaluations and less isolated post-update wall time than
restart. Because candidate placement is selected using clean behavior only,
neither this gate nor the boundary sweep changes the frozen obstacle-pair
denominator.

This pilot tests whether a late-arriving visual constraint can be handled
without discarding completed flow updates. It does not show that the original
sample was undecided, and it does not provide a formal safety guarantee. A
population claim that editability timing predicts the last safe correction
point requires independently selected obstacle scenes and predictions frozen
before their dynamic rollouts.

### 2.7 Registered executed-action early-exit pilot

This post-mechanistic amendment was frozen after the 15-state position grid but
before any early-exit action or rollout outcome. The position result shows that
the final layer/update has less causal transfer on positions 0--4, which are
executed before the next replan, than on deferred positions 5--9. That finding
does not itself establish that fewer flow evaluations preserve behavior. The
new pilot tests the consequence directly without retraining or changing the
public OpenPI velocity field.

After `k` evaluations on the unchanged public ten-step Euler time grid, output
the latest clean-action estimate `x_t - t v_t` and execute only its first five
controls before replanning. Every later replan uses the same rule. The primary
comparison is `k=7` versus the exact full-sampler control `k=10`; `k=7` is fixed
from the last boundary with complete donor control in the prior target curve,
not selected from early-exit behavior. The implementation substitutes the
ordinary integrated sampler output at `k=10`, making the control byte-exact even
though repeated float32 time subtraction can leave the algebraically equivalent
clean estimate about one ulp away. Boundary `k=4` is a frozen aggressive
sensitivity condition and cannot replace the primary comparison.

The first stage reuses all 16 serialized wine/bowl scene states and both task
directions with identical resets, observations, Gaussian noise by replan index,
and five-action execution horizon. Clean full-sampler competence and target-
first contact determine eligibility without inspecting early-exit outcomes; the
existing 15 dual-success scene clusters define the pilot population. Report
both directions but keep the physical scene as the inference cluster. A pilot
positive requires exact seven-versus-ten velocity-evaluation accounting at
every replan, lower isolated integration latency, and preservation of both
target-first contact and eventual task success in at least 14 of 15 clusters.
This threshold is descriptive and is not called noninferiority.

If the pilot passes, the confirmatory test uses the unchanged public LIBERO Goal
task order and 50 trials per task for 500 paired reset seeds under `k=7` and
`k=10`: 500 episodes per condition and 1,000 condition rollouts in total. A
paired loss occurs when the full sampler succeeds and early exit does
not. A practical positive requires the one-sided 95% Clopper--Pearson upper
bound on the paired loss probability to be below 0.02, exactly 30% fewer
velocity evaluations at every replan, and a scene-paired bootstrap interval for
median isolated integration-latency savings strictly above zero. Success with
an altered task denominator, adaptive boundary choice, or approximate compute
accounting is not accepted. This test complements public distillation and warm-
start approaches by evaluating zero-training reuse of the released sampler; it
does not claim the training-time gains of those methods.

The paired confirmation execution was frozen after the pilot decision but
before any suite-confirmation rollout. Each task/trial key has one serialized
reset and one deterministic Gaussian-noise stream indexed by replan; both
conditions must reproduce their hashes exactly. Within each task, condition
order is balanced 25/25 by the rank of a frozen SHA-256 digest of the
task/trial key, rather than chosen from behavior. A fixed unscored warm-up of
both conditions precedes timing in every new inference-server process. That
warm-up also requires the full-control physical action tensor to equal an
ordinary clean-sampler request exactly. No other client may use the policy GPU,
and each episode contributes one paired
success/loss outcome regardless of its number of replans. Every warm-up,
condition, pair, and progress artifact must name the same full Git commit, and
the launcher rejects tracked worktree changes. The registered 0.02 exact bound
permits at most four paired losses among 500 trials; five or more cannot pass. A
resume may skip only a pair whose two condition artifacts and hashes are
complete and whose code commit matches the current process.

The pooled gate is accompanied by reporting-only summaries for each of the ten
unchanged LIBERO Goal tasks: both condition success counts, paired losses and
gains, the exact paired-loss upper bound, the 25/25 condition-order audit, and
paired first-replan latency. These strata are always shown and are not used to
select tasks or create post hoc task-level pass criteria.

The observed pilot passes the descriptive rule: target-first contact is
preserved in 15/15 eligible clusters, eventual success and the registered
composite in 14/15, all replan compute counts are exact, and median paired
first-replan latency savings are 30.0%. The result opens the confirmation; it
does not alter its denominator, margin, pairing, or stopping rule.

## 3. Mechanistic hypotheses and utility amendment

The following hypotheses will be frozen after the pilot and before examining
confirmatory outcomes:

- **H1:** target identity loses conditional editability earlier in flow
  integration than grasp orientation.
- **H2:** trajectory homotopy or initial direction loses conditional
  editability earlier than gripper-closure timing.
- **H3:** layerwise target-identity mediation peaks earlier than geometric
  orientation mediation within an action-expert pass.
- **H4:** causal effects on future-token positions are temporally localized:
  early positions dominate initial direction, while positions near first
  contact dominate orientation and closure timing.
- **H5:** pi0.5 exhibits a sharper target-identity editability transition and
  better swap symmetry than matched pi0.

Recovery is initially exploratory because it requires closed-loop perturbation
episodes and may depend on task-specific observability.

The utility amendment adds:

- **U1:** the first-chunk editability boundary predicts the last successful
  continue-without-restart boundary on held-out scene states and beats the
  frozen fixed-boundary-seven rule in paired scene-cluster mean absolute error;
- **U2:** continuing at the preregistered safe boundary `k=7` is noninferior to
  full restart for new-target-first contact and eventual new-task success while
  using 70% fewer post-event velocity-field evaluations;
- **U3:** late negative-control boundaries `k>=9` reduce immediate correction
  even when subsequent clean replanning can sometimes recover eventual task
  success.
- **U4 (pilot):** in the selected moved-obstacle scene, at least one interrupted
  flow trajectory can be continued after a late visual update without collision
  or task loss and with fewer post-update velocity evaluations than restart.

## 4. Paired episode families

Each pair starts from an identical reconstructed physical transition state. Both
runs use identical preprocessing and Gaussian action noise. Only one registered
variable changes.

Candidate task pairs are discovered by canonicalizing the public BDDL scene and
then requiring the `obj_of_interest` and goal clauses to match after exactly one
atom substitution. The substituted atom's goal-argument position distinguishes
a manipulated-object change from a destination/subgoal change; these families
are never pooled. Public suite-level competence is established first, followed
by pair-level clean closed-loop validation.

### 4.1 Frozen pi0 competence gate

The matched pi0 checkpoint is evaluated only after the 30,000-update Orbax
checkpoint has finalized. OpenPI executes loop indices `0..29999`, increments
the serialized training state on every update, and keys the final save by the
zero-based loop index; the sole eligible directory is therefore `29999`, which
contains the state after 30,000 optimizer updates. Its three Orbax metadata and
manifest hashes are frozen in the executable checkpoint validator. Before any
pi0 intervention outcome is inspected, it
must pass both levels of the following gate:

1. The unchanged public OpenPI evaluator runs 50 episodes for each of the ten
   `libero_goal` tasks. The 500-episode success estimate must be at least 90%,
   and its Wilson 95% lower bound must be at least 87%.
2. On the 16 held-out wine-bottle-versus-bowl scene pairs with shared noise seed
   zero, at least 12 pairs must be dual-successful, restore exact inputs and
   simulator state, and contact the instructed object first in both directions.

The 90% suite threshold is fixed below the public pi0.5-LIBERO-goal reference
of 98.0% but high enough that causal differences are not dominated by a broadly
incompetent control. Because pi0.5 has 15/16 clean-eligible states, the
standalone 12/16 pi0 threshold mathematically guarantees only 11 common states.
Architecture-timing claims therefore require a separate minimum of 12 states
in the exact clean-eligible intersection before either pi0 intervention grid
can run. The clean-only selection artifact is written before this stop, so a
failed overlap and every contributing pair ID remain reportable. Primary
pi0-versus-pi0.5 estimates use only the passing intersection and retain every
exclusion. A competence-level failure is labeled competence-limited; a
common-state failure is labeled overlap-limited. Neither permits an
architectural timing comparison, and neither gate is relaxed or rescued by
selecting a later checkpoint.

Only a checkpoint passing both competence levels is converted to the hookable
PyTorch implementation with OpenPI's pinned public converter. The public
parameter mapping is called through the float32-intermediate safeguard proposed
in OpenPI PR #978: the intermediate model and saved checkpoint remain float32,
then OpenPI's unchanged policy loader recreates its intended mixed-precision
inference layout. This prevents an irreversible bfloat16 checkpoint downcast;
it does not change model weights, inputs, sampling, or the parity criterion.
Before any pi0
activation intervention, conversion parity is evaluated on both directions of
all 16 held-out target fixtures under identical seed-zero `50 x 32` action
noise. Every physical-action case must have maximum absolute JAX/PyTorch error
at most 0.02 and cosine similarity at least 0.999. A failed case blocks the
mechanistic comparison and is reported as conversion-limited; these tolerances
are not relaxed after conversion.

If and only if competence and conversion parity pass, both models are analyzed
on the intersection of their clean dual-success, instructed-target-first scene
states. Pi0 uses the same target manifest, seed-zero noise mode, ten flow
updates, evaluator thresholds, and 18 action-expert layer indices as pi0.5.
Its native action tensor remains `50 x 32`; it is not truncated before model
execution. The matched coarse grid crosses all ten flow steps and all 18 layers
with all action positions patched jointly, plus the same preregistered physical
action-dimension groups. The position grid crosses flow steps `{0, 7, 8, 9}`,
layers `{0, 8, 14, 17}`, and all 50 native pi0 positions.

Primary position comparisons use common positions zero through nine. A separate
normalized-chunk-time analysis partitions pi0.5's ten positions and pi0's 50
positions into ten fixed bins, averaging within bin and scene before the paired
model contrast. Every cross-model estimate is paired by serialized scene-state
hash. Timing comparisons report formation, editability-boundary, and
formation-to-editability-gap differences. Flow-shape comparisons report
retention AUC, late-weighting index, 10--90% transition width, and directional-
asymmetry AUC. Residual, action-state dimension, first-ten position, and
normalized-position cell differences use scene-cluster bootstrap intervals,
exact sign-flip tests when at most 20 clusters are eligible, and BH correction
within each metric family. This is a model-level comparison, not an isolated
architectural causal effect.

1. **Instruction target:** identical scene with two valid objects; change only
   the target phrase in the instruction.
2. **Target pose:** identical instruction and distractors; change only the
   designated target object's pose within a validated placement set.
3. **Obstacle position:** identical target and instruction; move one task-
   irrelevant obstacle between two validated positions.
4. **Recovery:** fork a common successful rollout immediately before a scripted
   perturbation; change only the perturbation state, then analyze the first
   post-perturbation chunk and subsequent closed-loop success.

Destination/subgoal pairs are evaluated from a phase-aligned post-grasp state,
not only from task initialization. For each direction, use the earliest state
starting five consecutive control steps in which the gripper contacts the common
manipulated object and the object is at least 2 cm above its initial height.
Restore that exact serialized state under both destination prompts and require
pixel-, wrist-image-, proprioception-, and simulator-state identity. Both the
base-derived and donor-derived snapshots are retained as separate blocks so the
result is not conditional on one destination's clean approach trajectory. The
2 cm lift, five-step persistence, and earliest-qualifying-state rule are frozen
before inspecting destination interventions.

Gripper-closure timing is evaluated from phase-aligned pre-contact states because
the initial target-pair chunks do not contain a closure event. For each clean
prompt direction, restore the state exactly one full 10-action horizon before
that rollout's first instructed-object contact. Retain both origin directions as
separate blocks only if neither registered target is already in gripper contact,
both prompts still first contact their instructed objects, and both tasks
succeed. A gripper property is eligible only when the clean first chunks have
different finite closure positions or one closes within the horizon while the
other does not. This selection uses clean behavior only.

If the prompt-swap pre-contact block is invalid because the unchanged object
layout causes cross-target contact, use a separately registered target-pose
family rather than relaxing the contact rule. Hold the instruction, robot state,
distractors, target height, and target orientation fixed, and translate only the
designated target object's planar free-joint coordinates along the clean
end-effector-to-target axis. The pilot offset grid is 2, 4, and 6 cm. Every
generalized position outside those two coordinates, every generalized velocity,
and every actuator state must remain bitwise identical. Candidate selection uses
only clean dual-success, first-target-contact, and first-chunk closure-contrast
criteria; patched outcomes are not inspected until the offset is frozen.

A pair enters causal analysis only if both unpatched endpoints are successful,
their measured property contrast exceeds the evaluator's minimum effect size,
and neither rollout violates simulator safety or validity checks. Exclusion
counts and reasons are always reported. Selection never uses patched outcomes.
An open-loop directional contrast alone is not evidence of target identity: the
instructed object must be the first task object contacted and the closed-loop
rollout must complete the task.

The pilot additionally reports a contact-valid sensitivity estimand: both sides
must exactly reproduce the saved initial input, simulator
state, and first clean chunk, and must first contact their instructed targets,
but a later placement failure does not remove the pair. The dual-success
estimand remains primary. This separation prevents a downstream placement
failure from being relabeled as failed target selection while making the
inclusion rule and its timing auditable. Confirmatory inclusion rules are frozen
before confirmatory interventions are inspected.

For the obstacle-position pilot, reuse a clean target-pair scene and treat the
non-instructed registered object as a movable distractor. Hold the prompt,
target pose, robot state, every non-obstacle generalized position, all
velocities, actuator state, obstacle height, and obstacle orientation fixed.
Move only the obstacle free joint's planar coordinates. Candidate centers are
placed at fractions `{0.35, 0.50, 0.65}` along the initial end-effector-to-target
line, with lateral offsets `{0.00, -0.05, +0.05}` meters in that fixed nested
order. Placements whose MuJoCo bounding spheres violate the frozen 1 cm
target-object or 4 cm gripper clearance are excluded before policy evaluation.

The initially attempted initialization state 0 yielded zero geometrically
valid placements, before any policy call or intervention outcome. The pilot
therefore expands across the existing 16 exact wine-bottle/bowl target-pair
states in their original manifest order (`init_index=0..15`). Within each state,
the frozen fraction/lateral grid order is unchanged. Geometry-invalid states
and clean-invalid placements remain in the denominator, and screening stops at
the first clean-eligible placement. No patched, dynamic-continuation, or timing
outcome may be read during this scan. The geometry and clean thresholds are not
relaxed.

That 16-state family was subsequently exhausted: all `16 x 9 = 144` placements
failed the same pre-policy target--obstacle bounding-sphere exclusion, and zero
policy outcomes were generated. Before screening any other obstacle scene, the
search expands to the already frozen public manipulated-object rows in
`catalogs/retarget_screening_plan.json`. The source-plan hash, exclusions, task
pairs, and initialization states are unchanged. To avoid spending the early
screen entirely on many initializations of one object contrast, the obstacle
plan uses a frozen task-diverse order: first occurrence defines target-pair
rank, then rows are ordered by `(init_index, target-pair rank)`. This order is
serialized and hashed before execution. The same nine placements and all
geometry and clean-behavior thresholds remain unchanged; the first clean-pass
rule and complete exclusion denominator are retained.

The first public-catalog row showed that its unchanged base episode contacted
the distractor first and failed. Because every obstacle placement in one source
row shares that exact base fixture, validating the base episode repeatedly
cannot change eligibility. Before processing the next source row, add an exact
source-base endpoint pre-gate: run the unchanged instructed-target side once,
require exact input/state restoration, target-first contact, and task success,
and skip all nine geometry/clean placements when it fails. This is logically
necessary for the existing paired-clean criterion and changes only compute;
the failed source row and reason remain in the denominator. No obstacle
intervention outcome is read.

Multiple target-pair rows can share an identical base task, initialization, and
serialized base fixture. After observing repeated alphabet-soup base endpoints
but before the next duplicated base family completes, cache the source-base
gate only when image, wrist image, proprioception, simulator state, and prompt
signatures are all exact. Every catalog row retains its own exclusion record
and points to the reused gate artifact and digest. A signature mismatch forces
a fresh rollout. This changes compute only, not the row order, denominator, or
eligibility rule.

Selection then uses clean closed-loop behavior only. Both paired rollouts must
restore their fixture exactly, contact the unchanged instructed target first,
and complete the task. Over the first five executed controls, the unmodified
trajectory's counterfactual planar clearance to the moved obstacle must be at
most the obstacle bounding radius plus 2 cm; the obstacle-scene trajectory must
avoid obstacle contact, clear the bounding radius, improve center clearance by
at least 1.5 cm, and differ in its five-step end-effector endpoint by at least
1 cm. Select the first passing placement in the registered order and retain all
failed rows. This creates a clean obstacle-sensitive pilot without selecting on
patched outcomes. It does not yet establish that a late-arriving safety update
can redirect the robot; that requires a separate physical-state-aligned dynamic
condition-update experiment.

Eligibility is property-specific. For example, an instruction pair whose clean
chunks both keep the gripper open may be valid for target direction but is not
evidence about gripper-closure editability. Normalized effects are never
interpreted when their clean endpoint denominator is below the frozen pilot
threshold.

## 5. Intervention families

### 5.1 Flow switch

For every switch boundary from zero through all Euler steps, integrate the same
initial noise under the base prefix before the boundary and the donor prefix
afterward. Cache keys and values are switched as a complete conditioning object.

Categorical target identity is tested online, not inferred from action distance:
the same boundary intervention is applied at every replan until the first task-
object contact, at which point the rollout terminates. The outcome is source-
target retention, donor-target transfer, or contact with neither target. Full-
donor and full-source boundaries are required positive and identity controls.
A separate first-replan-only experiment measures whether later clean replans
recover from an intervened initial chunk; it is labeled recovery rather than
pooled into the repeated-intervention target-identity curve.

Categorical destination identity is evaluated from the phase-aligned held-object
fixtures. The same boundary intervention is repeated until the released object
has remained within 8 cm of either registered destination for five consecutive
steps, or the source task succeeds. The categorical outcome is the registered
destination nearest the final live MuJoCo object position. The radius and
persistence rule affect termination only; endpoint identity is always scored by
the two registered distances. Full-donor placement, full-source placement, and
both clean endpoint margins must pass before intermediate boundaries are
interpreted.

### 5.2 Residual-stream patch

Patch the post-layer action-expert residual stream at one `(s, l)` site. The
coarse screen patches all future positions together. Follow-up experiments patch
single positions or preregistered temporal windows only at coarse-screen peaks.

### 5.3 Path patch

At selected peaks, patch attention and MLP contributions separately to
distinguish conditioning transfer from within-chunk computation. These analyses
are secondary and must reproduce the total residual-patch direction.

### 5.4 Action dimensions

Action dimensions are not hidden-state coordinates. Dimension interventions
therefore operate on `x_t`, `v_t`, or the decoded action tensor, not arbitrary
residual channels. The primary groups are translation, rotation, and gripper;
robot-specific auxiliary dimensions are reported separately. Scalar-dimension
results are exploratory unless a robot's action semantics justify them a priori.

## 6. Outcome evaluators

Evaluators are deterministic functions versioned with the experiment.

- **Target identity:** signed terminal approach/grasp affinity to the two target
  objects, plus categorical first-contact identity in rollout. The initial
  direct-line Cartesian projection is retained only as a failed pilot proxy and
  is not a categorical target-identity gate.
- **Trajectory direction:** initial end-effector displacement projected onto the
  clean base-to-donor trajectory contrast; for obstacles, the signed homotopy
  class around the obstacle. This contrast need not point directly at either
  target when collision avoidance or shared approach geometry intervenes.
- **Grasp orientation:** geodesic wrist-orientation error relative to each
  candidate grasp frame, evaluated over a preregistered pre-contact window.
- **Gripper closure:** first future position crossing a calibrated closure
  threshold, with right censoring if closure is absent from the chunk.
- **Recovery:** first post-perturbation chunk direction, time to resume progress,
  and eventual closed-loop task success.

Thresholds and contact windows are calibrated on pilot demonstrations without
viewing intervention effects, then frozen.

Rollout phase anchors begin with Event-SAE's pinned AWE position-only dynamic
program and a pilot error-threshold sensitivity sweep. Its geometric-gripper
mode is not used unless the logged policy command is demonstrably discrete;
continuous OpenPI gripper values make exact command-change detection degenerate.

## 7. Effect estimates

For continuous vector-valued outcomes, normalized causal transfer is

```text
NCTE = dot(Y_patch - Y_A, Y_B - Y_A) / ||Y_B - Y_A||^2.
```

Zero is the base behavior and one is full donor transfer. Values outside this
range are retained rather than clipped. For categorical outcomes, report the
change in donor-class probability from a calibrated evaluator and the hard flip
rate. Every estimate is computed in both patch directions.

Noise seeds are repeated measurements, not independent experimental units.
They are averaged within a scene pair for nonparametric summaries and modeled
as nested repeated effects in sensitivity analyses.

Pairs that reuse the same serialized scene state with different target
contrasts are also dependent. Their simulator-state hash defines an outer
cluster for resampling or mixed-effects analysis; they are never counted as
independent scene pairs merely because their prompt pair differs.

## 8. Controls and falsification tests

- **Identity patch:** `A -> A` must reproduce the unpatched action within the
  recorded numerical tolerance.
- **Full donor ceiling:** replacing the complete relevant state must approach
  the donor endpoint.
- **Unrelated donor:** a matched but property-irrelevant donor should not produce
  systematic directed transfer.
- **Direction symmetry:** `A -> B` and `B -> A` effects must agree after sign
  alignment; asymmetry is reported, not averaged away.
- **Seed determinism:** repeated unpatched runs with saved noise must be bitwise
  equal when deterministic kernels permit, otherwise within a frozen tolerance.
- **No-op hook:** enabling capture without patching must not alter outputs.
- **Distribution check:** activation norms and downstream numerical stability
  are compared with clean runs to detect pathological hybrid states.
- **Behavioral validity:** causal claims are conditional on competent clean
  behavior; failures are analyzed separately rather than silently discarded.

## 9. Sampling and analysis

The pilot uses 16 valid scene pairs per counterfactual family and four shared
noise seeds. Pilot data choose evaluator thresholds, temporal windows, batching,
and the subset of layer/step peaks; they are never pooled into confirmation.

The initial confirmatory target is 50 valid scene pairs per retained family and
four shared noise seeds per pair. Before confirmation, a simulation-based power
analysis using only pilot variance may increase this number. It may not reduce
the sample below 50 or use the observed confirmatory effect.

Primary intervals use a scene-pair cluster bootstrap. Ordered editability
hypotheses use the bootstrap distribution of paired boundary differences.
Exploratory heatmap cells use Benjamini-Hochberg false-discovery-rate correction
within each property and intervention family. Raw effects, uncertainty, and
effective sample sizes accompany every heatmap.

The pi0 comparison uses the same accepted pair manifest, noise tensors,
evaluators, and intervention grid. Model is a fixed effect and scene pair a
shared random effect. Because pretraining differs, comparisons are described as
model-level contrasts, not pure architectural causation.

## 10. Staged execution and stopping rules

1. Reproduce the published pi0.5-LIBERO baseline within its documented
   evaluation protocol.
2. Validate one instruction-target pair end to end.
3. Pass every no-op, identity, determinism, and full-donor control.
4. Run the 16-pair pilot and freeze evaluators and confirmatory families.
5. Run pi0.5 confirmation without changing hypotheses or thresholds.
6. Train/evaluate matched pi0 and run the identical accepted manifest.
7. Add recovery only after static-chunk controls pass.

If full-donor controls cannot transfer the intended property, the corresponding
pair family is not interpretable and is redesigned before confirmation. If
identity/no-op controls exceed tolerance, causal sweeps stop until the
instrumentation is corrected.

## 11. Required release artifacts

- exact OpenPI and LIBERO commits;
- checkpoint URIs and content hashes;
- environment lock and hardware/software manifest;
- accepted and excluded pair manifests with reasons;
- saved Gaussian noise tensors or deterministic generators;
- raw clean and patched outcome tables;
- intervention configuration and hook-site schema;
- analysis scripts, figure sources, confidence intervals, and negative results;
- a limitations section separating internal causal effects from claims about
  cognition or human-like planning.
