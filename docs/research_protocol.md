# Research protocol: causal commitment in action chunks

Protocol version: 0.1 (design-stage preregistration)

## 1. Research question

When does a flow-matching vision-language-action model become causally committed
to the semantic and geometric properties of a predicted action chunk?

The primary model is the public pi0.5-LIBERO checkpoint. A pi0 model fine-tuned
on the same demonstrations, preprocessing, optimization budget, and evaluation
scenes is the matched control. Pi0.7 is outside the confirmatory scope until its
weights and internal implementation are public.

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

### 2.3 Flow-step commitment

Commitment is measured with a suffix conditioning switch. The action state is
integrated using condition `A` for the first `k` Euler updates and condition `B`
for all remaining updates. Let `Y` be a continuous property score oriented from
`A` to `B`. Retention is

```text
R_z(k) = 1 - (Y_switch(k) - Y_A) / (Y_B - Y_A).
```

The symmetric estimate averages `A -> B` and `B -> A` after orienting both to
retention of the initial condition. The commitment step is the earliest `k`
whose isotonic mean retention is at least 0.8 and remains at least 0.8 for every
later switch. Confidence intervals are cluster-bootstrapped by scene pair.

Because a threshold crossing alone can obscure whether influence accumulates
uniformly across Euler updates, every curve is also compared with the
`R(k)=k/S` uniform-step null. Report retention AUC, half-commitment boundary,
and the marginal retention contributed by the final update. A positive
`0.5 - AUC` is a late-weighting index, not by itself evidence for a discrete
internal planning phase.

This is commitment relative to a natural, in-distribution counterfactual—not a
claim that no arbitrarily large perturbation could ever alter the output.

## 3. Confirmatory hypotheses

The following hypotheses will be frozen after the pilot and before examining
confirmatory outcomes:

- **H1:** target identity commits earlier in flow integration than grasp
  orientation.
- **H2:** trajectory homotopy or initial direction commits earlier than
  gripper-closure timing.
- **H3:** layerwise target-identity mediation peaks earlier than geometric
  orientation mediation within an action-expert pass.
- **H4:** causal effects on future-token positions are temporally localized:
  early positions dominate initial direction, while positions near first
  contact dominate orientation and closure timing.
- **H5:** pi0.5 exhibits a sharper target-identity commitment transition and
  better swap symmetry than matched pi0.

Recovery is initially exploratory because it requires closed-loop perturbation
episodes and may depend on task-specific observability.

## 4. Paired episode families

Each pair starts from an identical serialized simulator and robot state. Both
runs use identical preprocessing and Gaussian action noise. Only one registered
variable changes.

1. **Instruction target:** identical scene with two valid objects; change only
   the target phrase in the instruction.
2. **Target pose:** identical instruction and distractors; change only the
   designated target object's pose within a validated placement set.
3. **Obstacle position:** identical target and instruction; move one task-
   irrelevant obstacle between two validated positions.
4. **Recovery:** fork a common successful rollout immediately before a scripted
   perturbation; change only the perturbation state, then analyze the first
   post-perturbation chunk and subsequent closed-loop success.

A pair enters causal analysis only if both unpatched endpoints are successful,
their measured property contrast exceeds the evaluator's minimum effect size,
and neither rollout violates simulator safety or validity checks. Exclusion
counts and reasons are always reported. Selection never uses patched outcomes.

Eligibility is property-specific. For example, an instruction pair whose clean
chunks both keep the gripper open may be valid for target direction but is not
evidence about gripper-closure commitment. Normalized effects are never
interpreted when their clean endpoint denominator is below the frozen pilot
threshold.

## 5. Intervention families

### 5.1 Flow switch

For every switch boundary from zero through all Euler steps, integrate the same
initial noise under the base prefix before the boundary and the donor prefix
afterward. Cache keys and values are switched as a complete conditioning object.

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
  objects, plus categorical contacted-object identity in rollout.
- **Trajectory direction:** initial end-effector displacement projected onto the
  base-to-donor contrast; for obstacles, the signed homotopy class around the
  obstacle.
- **Grasp orientation:** geodesic wrist-orientation error relative to each
  candidate grasp frame, evaluated over a preregistered pre-contact window.
- **Gripper closure:** first future position crossing a calibrated closure
  threshold, with right censoring if closure is absent from the chunk.
- **Recovery:** first post-perturbation chunk direction, time to resume progress,
  and eventual closed-loop task success.

Thresholds and contact windows are calibrated on pilot demonstrations without
viewing intervention effects, then frozen.

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

Primary intervals use a scene-pair cluster bootstrap. Ordered commitment
hypotheses use the bootstrap distribution of paired commitment-step differences.
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
