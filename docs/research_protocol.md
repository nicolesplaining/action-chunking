# Research protocol: causal editability and retargeting in action chunks

Protocol version: 0.2 (construct-validity amendment before utility outcomes)

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

For each accepted state, the separately measured first-chunk editability
boundary predicts the last continued boundary that preserves new-target-first
contact and eventual task success. Association is reported with paired
state-level accuracy and Spearman correlation; the useful claim requires
out-of-sample scene states and cannot be established from the original 15
mechanistic states alone.

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
  continue-without-restart boundary on held-out scene states;
- **U2:** continuing at the preregistered safe boundary `k=7` is noninferior to
  full restart for new-target-first contact and eventual new-task success while
  using 70% fewer post-event velocity-field evaluations;
- **U3:** late negative-control boundaries `k>=9` reduce immediate correction
  even when subsequent clean replanning can sometimes recover eventual task
  success.

## 4. Paired episode families

Each pair starts from an identical serialized simulator and robot state. Both
runs use identical preprocessing and Gaussian action noise. Only one registered
variable changes.

Candidate task pairs are discovered by canonicalizing the public BDDL scene and
then requiring the `obj_of_interest` and goal clauses to match after exactly one
atom substitution. The substituted atom's goal-argument position distinguishes
a manipulated-object change from a destination/subgoal change; these families
are never pooled. Public suite-level competence is established first, followed
by pair-level clean closed-loop validation.

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
