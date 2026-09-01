# Manuscript claim and evidence outline

## Working title

**When can a VLA action sample still be retargeted? Causal editability of
flow-matching action chunks**

## Central question

When can a flow-matching VLA action sample still be redirected across denoising
time, transformer depth, and future action-token position, and does that
boundary enable useful correction without restarting generation?

The study operationalizes three distinct quantities:

1. **Formation:** the earliest clean flow estimate whose paired contrast stays
   within the frozen relative-error tolerance of the final contrast.
2. **Mediation:** the causal transfer produced by an interchange intervention at
   a specified flow step, action-expert layer, action-token position, or action
   dimension.
3. **Conditional editability:** the earliest conditioning-switch boundary whose
   isotonic source-retention curve remains above the frozen 0.8 threshold.

The third quantity does not diagnose indecision or absence of an internal plan.
It measures whether natural counterfactual conditioning still has causal control
over the sample. Its value is tested independently by dynamic retargeting under
the newly instructed task and subsequent clean replanning.

This distinction builds on pi0/pi0.5 flow-matching action generation
[@black2024pi0; @physicalintelligence2025pi05], action chunking
[@zhao2023act], diffusion-style receding-horizon control
[@chi2023diffusionpolicy], and causal interchange interventions
[@geiger2022iit; @geiger2023causalabstraction].

## Current evidence table

| Claim scope | Registered contrast | Evidence | Allowed claim |
| --- | --- | --- | --- |
| Population pilot | Wine bottle versus bowl target; 15 serialized scene states, seed 0 | Formation step 0 for full action, translation, and rotation; editability boundaries 10, 10, and 9; positive within-state gap in all 15 states | Geometric contrasts can be legible at the first flow estimate while remaining causally editable until the final updates |
| Population pilot | Same 15 states; repeated online intervention | 30/30 monotonic directional curves; target editability boundary median 8, IQR 8--9; exact donor/source controls | Target identity remains redirectable through 70% of integration and loses donor control during the last 20--30% |
| Population pilot | Same 15 states; layer grid | Final-layer, final-flow residual peaks near NCTE 0.095--0.099 with clustered intervals excluding zero and BH-adjusted q below `7.1e-5` | A reproducible late residual mediator exists, but residual patches are not a complete causal decomposition |
| Population pilot | Same 15 states; 4 flow steps x 4 layers x 10 future positions | At the final layer/update, positions 5--9 exceed positions 0--4 for full action by NCTE 0.0110 (95% interval 0.0101--0.0119; exact sign-flip p=6.1e-5), with the same sign in 15/15 states | Late residual computation preferentially affects deferred rather than immediately executed controls |
| Population pilot | Same 15 states; grouped action-state grid | Step-9 `x_t` translation-to-translation NCTE 0.537 and rotation-to-rotation NCTE 0.900 | Late action-state coordinates instantiate substantial property-specific geometry |
| Replicated narrow pilot | Bowl-on-stove versus bowl-on-plate after grasp; two independently selected physical-state blocks | Four monotonic directions; editability boundaries 7 and 8 in both blocks | A post-grasp destination/subgoal can be causally transferred until around 70--80% of integration |
| Conditional pilot | Wine-bottle target pose, one state, seed 0 | Closure formation and editability boundary at 7; only token 9 affects categorical closure in 120 token sites | Closure timing can be localized to its matching future token for one sampled action mode |
| Negative robustness result | Same target-pose state, four noise seeds | Only seed 0 has a closure contrast; all-seed eligible state count is zero | The closure-token result is noise conditional and is not a state-level population effect |
| Negative recovery pilot | Alphabet soup versus cream cheese, first-replan donor chunk | Exact inputs, simulator restoration, swapped chunks, and clean controls; 0/2 directions have an intervention-induced donor-first contact | Eventual success is not interpretable as recovery unless the patched chunk first causes the defined failure event |
| Positive inference pilot | Same 16-state wine/bowl manifest; 15 clean-eligible scene clusters; seven versus ten evaluations at every replan | Target-first contact preserved in 15/15 clusters; eventual success and composite preserved in 14/15; exact 30% evaluation savings; median latency savings 30.0% (95% bootstrap interval 27.4--32.9%) | The executed five-control prefix can often be emitted three evaluations early, but this exploratory same-population pilot is not confirmatory noninferiority or evidence of indecision |
| Positive matched-control gate | Registered 30,000-update pi0; 500 LIBERO Goal episodes and 16 held-out target pairs | 465/500 suite success (Wilson 95% interval 0.904--0.949); 15/16 exact dual-success target-first pairs | Pi0 is behaviorally competent enough for an architecture control |
| Negative conversion gate | Same pi0 JAX checkpoint converted with OpenPI's default bfloat16 output; 32 exact held-out action cases | 24/32 pass; maximum error 2.013; minimum cosine 0.806; no intervention launched | The first converted runtime is not numerically matched and cannot support architecture-level claims |

## Primary results narrative

### R1. Clean formation precedes loss of conditional editability

Use the 15-state target-pair cluster bootstrap and paired timing-gap analysis.
Lead with the full-action and translation mean gap of 9.53 flow steps (95%
interval 9.27--9.80), followed by rotation 8.80 (8.60--9.00) and target
direction 8.20 (6.80--9.40). Report that all eligible state-level gaps are
positive. Do not equate early decodability or clean contrast alignment with
loss of counterfactual control.

### R2. Behavioral target identity remains conditionally editable late

Report the complete 15-state online categorical curve. Source retention is zero
through boundary 7, 0.600 at boundary 8 (cluster-bootstrap interval
0.433--0.767), and one at boundaries 9 and 10. Eighteen directional curves
cross the editability threshold at 8 and 12 at 9. Preserve the one boundary-1 neither-target outcome in
the raw table.

### R3. Geometry is instantiated late in the action state and final layer

Report all raw cells, clustered intervals, directional asymmetry, exact or
Monte Carlo sign-flip p-values, and BH q-values. The property-matched `x_t`
effects are the strongest results. The final residual-layer peaks are smaller
but replicated. Avoid describing the final layer as the unique planning layer:
many earlier sites are FDR positive, and interchange effects are not additive.

### R4. Subgoal timing replicates across phase-aligned physical states

The destination curve is a post-grasp estimand, not directly exchangeable with
initial-scene target timing. Its two physical-state blocks independently produce
the same step function. Present it as evidence that editability timing depends on
rollout phase and property, not as proof that subgoals universally precede or
follow object choice.

### R5. Gripper timing is identifiable but noise conditional

Present the exact token-9 localization as a mechanistic case study, alongside
the four-noise exclusion result. The conservative all-seed gate prevents the
one eligible seed from becoming a population estimate. The next experiment
must expand target-pose pairs across scene states and report the fraction of
noise modes with an eligible closure contrast.

### R6. Recovery requires an intervention-induced failure event

Retain the first-replan endpoint pilot as a negative result. Although all
rollouts eventually succeeded and every exactness control passed, neither
direction met the frozen recovery eligibility gate. One tempting direction had
the same wrong-object-first behavior under its identity control, so the donor
chunk did not cause the event from which recovery would be inferred. Do not
report an eventual recovery rate for this pair; use it to motivate clean-screened
failure induction before any population recovery timing study.

### R7. Utility must be shown by retargeting without restart

Use held-out scene states and treat the new instruction as the actual task goal.
After `k` old-instruction flow updates, either continue from `x_k` under the new
instruction or discard the sample and restart from the same original noise.
Compare new-target-first contact, full task success, subsequent clean replanning,
post-event velocity evaluations, and synchronized latency. The useful claim is
allowed only if the measured editability boundary predicts the last successful
continue boundary and `k=7` is noninferior to restart while using three rather
than ten post-event velocity evaluations.

Add the boundary-adaptive policy as the strongest practical test. For an
instruction update at flow step `k`, continue only if the frozen scene-specific
predictor says `k` remains recoverable; otherwise restart. Invalid predictions
restart. Compare this policy with always restart and a fixed global cutoff at
seven, averaging update times 0--10 uniformly within each independent scene.
Require target-first contact, task success, and their composite to be
noninferior to both controls, and require cluster-bootstrap velocity-evaluation
savings intervals above zero against both. Describe the uniform update-time
average as a controlled design estimand rather than a deployment distribution.

Require a separate final audit to reconstruct every cluster from its raw sweep
summary, 12-condition rollout table, and grasp-orientation artifact, recheck all
frozen input hashes, and rerun every statistic. A utility result is reportable
only when that independently rebuilt payload exactly matches the saved summary.

Position this as training-free reuse of an interrupted flow trajectory, not as
a general steering algorithm. DynaGuide and Guided Action Flow add learned
guidance during iterative generation; DSRL learns a latent-noise policy; STEP
learns a warm-start predictor; Consistency Policy distills the sampler. Our
question is narrower and complementary: when the pretrained VLA already
understands both instructions, can its measured conditional-editability region
identify which completed velocity evaluations can be safely retained after the
instruction changes?

Treat A1 and FASTER as the closest efficient-VLA comparisons. A1 learns
intermediate-layer exits and warm-starts truncated flow matching across VLM
layers; FASTER changes the flow schedule to prioritize near-term action
positions and streams execution. The present intervention leaves released
pi0.5 weights, the uniform schedule, and the ordinary flow path unchanged. Its
novelty claim is causal selection of an executable prefix followed by sealed
closed-loop validation, not early exit or fast flow sampling in isolation.

Decompose failures by whether the old event occurs in the five intervened
actions, the first registered contact occurs only after a later clean replan,
or task failure follows correct target-first contact. Report the first-contact
replan-index histogram rather than collapsing all failures into eventual
success alone.

The confirmatory population expands in the frozen public-catalog order until 59
independent endpoint-eligible scene clusters are available or the catalog is
exhausted, regardless of whether the 16-state pilot is eligible. Select the first
endpoint-eligible direction in frozen gate order within each cluster and retain
all additional directions only in the audit trail. Freeze every selected
action-only prediction before the first continuation rollout in that population.
Report exact and within-one-boundary prediction accuracy, mean absolute error,
Spearman correlation, success at the predicted boundary, and failure at its
successor. For boundary seven, report the one-sided exact upper confidence bound
on paired composite losses, with one scene cluster per trial, alongside median
paired latency savings and its frozen cluster-bootstrap interval.

For execution only, screen two adjacent outcome-independent catalog rows on
isolated GPU/port workers. Preserve the inferential population as the exact
contiguous frozen prefix; retain and label at most one concurrently completed
successor as speculative and exclude it from selection. Do not reuse the prior
28-row exploratory screen because it lacks a code-commit binding.

The first state-zero pilot gives a qualified positive result. At `k=7`, both
directions reached the newly instructed object first and completed the new task,
matching restart while using 3/10 post-event velocity evaluations. Median
microbenchmark latency fell from 378.95 ms to 191.67 ms; closed-loop mean
post-event latency fell from 400.84 ms to 179.69 ms. All exactness controls
passed. But `k=8,9,10` also succeeded in both directions, including `k=10`,
which performs no new-instruction velocity evaluation. Subsequent clean
replanning rescued a benign first chunk, so this initial-state pilot supports
an inference-efficiency demonstration but not the claim that the measured
boundary predicts the last successful recovery point. Require a pre-contact
failure-induction gate before the held-out test.

### R8. Late deferred-token mediation motivates an inference shortcut

Report the post-grid early-exit pilot as a distinct practical result, not as the
recovery test. Seven evaluations preserve correct target-first contact in all
15 eligible scene clusters and the target-first-plus-eventual-success composite
in 14, with exact 30% evaluation savings. Separate the one eventual-success
loss from instruction following: both conditions contact the instructed bowl
first at step 35, but only the ten-evaluation control finishes. Report the wide
14/15 exact interval, all 15 paired latency differences, and the fixed-order
limitation. The pilot reuses the mechanistic population and therefore opens but
does not replace the frozen confirmation on 500 episode pairs (1,000 condition
rollouts).

## Matched pi0 control

The control uses the public pi0 base architecture and upstream `pi0_libero`
full-finetuning recipe at the pinned OpenPI revision. It uses the same public
LIBERO dataset, newly computed finite normalization statistics, 30,000 update
steps, batch size 32, two-device FSDP, and EMA 0.99. After baseline competence
is established, reuse the accepted manifests, saved Gaussian noise, evaluator
thresholds, intervention sites, and scene-state clustering.

Run the coarse all-flow-step/all-layer grid and the population action-position
grid only on the clean-eligible scene intersection. Preserve pi0's native
50-position action tensor. Compare positions zero through nine directly and use
ten frozen normalized-chunk-time bins as a sensitivity analysis. Report paired
scene-state differences in formation, editability, formation-to-editability gap,
retention AUC, 10--90% transition width, directional-asymmetry AUC, and common
causal cells with within-metric BH correction.

The two-device pipeline passed its execution smoke and completed the frozen
30,000-update run. The final checkpoint passes both behavioral gates: 465/500
LIBERO Goal episodes and 15/16 exact dual-success target pairs. Its first
bfloat16 conversion fails parity at 24/32 cases, so the matched mechanistic
comparison remains conversion-limited. Rerun the same cases with the public
float32-intermediate repair and unchanged thresholds; only a 32/32 pass may
release the two-H100 coarse and all-50-position grids.

The model comparison is a matched experimental protocol, not a pure
architectural ablation: pi0 and pi0.5 differ in pretraining and checkpoint
history. Model-by-property contrasts must be reported with that limitation.

## Obstacle-position extension

Move only a task-irrelevant distractor's planar free-joint coordinates onto a
fixed grid around the clean end-effector-to-target corridor. Freeze geometric
exclusions and clean behavioral eligibility before interventions. The selected
pair must preserve exact restoration, target-first dual success, and first-chunk
obstacle avoidance while the unmodified trajectory would intersect the moved
obstacle corridor. Interpret its causal grid as obstacle-sensitive trajectory
editability, not yet as successful response to a late safety constraint.

State 0 had no geometrically valid placement and generated no policy outcome.
The registered expansion scans the original 16 exact target-pair states in
manifest order, preserving the grid, all exclusions, and the first-clean-pass
stop rule. Report state 0 and every later screened row in the denominator.

All 144 wine/bowl placements were ultimately geometry-invalid, with no policy
call. The broader search therefore reuses the frozen public manipulated-object
catalog in a serialized task-diverse round-robin order while changing no
placement or eligibility threshold. Retain the complete wine/bowl and broader
catalog denominators in the supplement.

The registered dynamic follow-up always executes in the moved-obstacle world.
It compares restart with continuation from every interrupted flow state after
switching from the paired original-pose image to the live obstacle image. Call
this a consequential safety result only if the old-condition chunk causes the
registered collision, restart avoids it and completes the task, and at least
one `k>0` continuation also avoids it and succeeds with fewer post-update
velocity evaluations. Report the full boundary curve and label a one-scene
result as a pilot, not a prediction claim or formal safety guarantee.

## Required figures

The current figure build is generated directly from immutable CSV tables by
`scripts/make_manuscript_figures.py`; `figure_manifest.json` records every input
path and SHA-256.

1. Formation error, continuous action retention, and closed-loop target
   retention in one aligned flow-time figure.
2. Flow-step by residual-layer heatmaps plus property-matched `x_t` and `v_t`
   coordinate effects. Open circles require both BH-adjusted `q < 0.05` and a
   scene-cluster interval strictly above zero; negative transfer remains visible
   in the color field but is not counted as positive mediation.
3. Target-identity and replicated post-grasp destination curves on shared axes.
4. The token-9 closure case, labeled as descriptive `n=1` eligible
   state/noise-mode evidence with no population significance marker.
5. Paired pi0 versus pi0.5 editability and mediation differences after the pi0
   competence gate passes.
6. A population token-position figure only after the 15-state token sweep is
   explicitly authorized and complete.

## Explicit nonclaims

- Conditional editability under a natural paired counterfactual is not robustness to an
  arbitrary perturbation.
- Successful late conditioning redirection is not evidence that the policy was
  undecided or lacked an internal plan; it may be strong instruction following.
- Activation patching does not establish a human-like symbolic plan.
- Different rollout phases cannot be ordered into a universal semantic
  hierarchy without a phase-matched design.
- A single eligible noise seed is not a replicated gripper result.
- Eventual success after a patched chunk is not recovery unless the patch
  induces the preregistered failure event relative to the identity control.
- Probe accuracy or clean decodability is not causal editability.
- The current pi0.5 result does not determine whether pi0 or pi0.7 has the same
  timing; pi0 requires the matched control run, and public pi0.7 internal weights
  are unavailable.
