# Manuscript claim and evidence outline

## Working title

**Formation is not commitment: causal timing of action chunks in a
flow-matching vision-language-action model**

## Central question

When does a flow-matching VLA become causally committed to target choice,
subgoal, geometry, and gripper behavior across denoising time, transformer
depth, and future action-token position?

The study operationalizes three distinct quantities:

1. **Formation:** the earliest clean flow estimate whose paired contrast stays
   within the frozen relative-error tolerance of the final contrast.
2. **Mediation:** the causal transfer produced by an interchange intervention at
   a specified flow step, action-expert layer, action-token position, or action
   dimension.
3. **Commitment:** the earliest conditioning-switch boundary whose isotonic
   source-retention curve remains above the frozen 0.8 threshold.

This distinction builds on pi0/pi0.5 flow-matching action generation
[@black2024pi0; @physicalintelligence2025pi05], action chunking
[@zhao2023act], diffusion-style receding-horizon control
[@chi2023diffusionpolicy], and causal interchange interventions
[@geiger2022iit; @geiger2023causalabstraction].

## Current evidence table

| Claim scope | Registered contrast | Evidence | Allowed claim |
| --- | --- | --- | --- |
| Population pilot | Wine bottle versus bowl target; 15 serialized scene states, seed 0 | Formation step 0 for full action, translation, and rotation; commitment 10, 10, and 9; positive within-state gap in all 15 states | Geometric contrasts can be legible at the first flow estimate while remaining causally editable until the final updates |
| Population pilot | Same 15 states; repeated online intervention | 30/30 monotonic directional curves; target commitment median 8, IQR 8--9; exact donor/source controls | Target identity remains redirectable through 70% of integration and fixes during the last 20--30% |
| Population pilot | Same 15 states; layer grid | Final-layer, final-flow residual peaks near NCTE 0.095--0.099 with clustered intervals excluding zero and BH-adjusted q below `7.1e-5` | A reproducible late residual mediator exists, but residual patches are not a complete causal decomposition |
| Population pilot | Same 15 states; grouped action-state grid | Step-9 `x_t` translation-to-translation NCTE 0.537 and rotation-to-rotation NCTE 0.900 | Late action-state coordinates instantiate substantial property-specific geometry |
| Replicated narrow pilot | Bowl-on-stove versus bowl-on-plate after grasp; two independently selected physical-state blocks | Four monotonic directions; commitment boundaries 7 and 8 in both blocks | A post-grasp destination/subgoal can be causally transferred and fixes around 70--80% of integration |
| Conditional pilot | Wine-bottle target pose, one state, seed 0 | Closure formation and commitment at 7; only token 9 affects categorical closure in 120 token sites | Closure timing can be localized to its matching future token for one sampled action mode |
| Negative robustness result | Same target-pose state, four noise seeds | Only seed 0 has a closure contrast; all-seed eligible state count is zero | The closure-token result is noise conditional and is not a state-level population effect |

## Primary results narrative

### R1. Clean formation precedes causal commitment

Use the 15-state target-pair cluster bootstrap and paired timing-gap analysis.
Lead with the full-action and translation mean gap of 9.53 flow steps (95%
interval 9.27--9.80), followed by rotation 8.80 (8.60--9.00) and target
direction 8.20 (6.80--9.40). Report that all eligible state-level gaps are
positive. Do not equate early decodability or clean contrast alignment with an
irreversible plan.

### R2. Behavioral target identity fixes in the last integration quarter

Report the complete 15-state online categorical curve. Source retention is zero
through boundary 7, 0.600 at boundary 8 (cluster-bootstrap interval
0.433--0.767), and one at boundaries 9 and 10. Eighteen directional curves
commit at 8 and 12 at 9. Preserve the one boundary-1 neither-target outcome in
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
the same step function. Present it as evidence that commitment timing depends on
rollout phase and property, not as proof that subgoals universally precede or
follow object choice.

### R5. Gripper timing is identifiable but noise conditional

Present the exact token-9 localization as a mechanistic case study, alongside
the four-noise exclusion result. The conservative all-seed gate prevents the
one eligible seed from becoming a population estimate. The next experiment
must expand target-pose pairs across scene states and report the fraction of
noise modes with an eligible closure contrast.

## Matched pi0 control

The control uses the public pi0 base architecture and upstream `pi0_libero`
full-finetuning recipe at the pinned OpenPI revision. It uses the same public
LIBERO dataset, newly computed finite normalization statistics, 30,000 update
steps, batch size 32, two-device FSDP, and EMA 0.99. After baseline competence
is established, reuse the accepted manifests, saved Gaussian noise, evaluator
thresholds, intervention sites, and scene-state clustering.

The two-device pipeline has passed a two-update execution smoke from the public
base checkpoint (first reported loss 0.1812; finite gradient and parameter
norms; finalized Orbax checkpoint). This validates the pipeline only. The pi0
comparison remains pending until the 30,000-update checkpoint passes the frozen
LIBERO competence gate.

The model comparison is a matched experimental protocol, not a pure
architectural ablation: pi0 and pi0.5 differ in pretraining and checkpoint
history. Model-by-property contrasts must be reported with that limitation.

## Required figures

1. Formation and commitment curves with clustered intervals for four eligible
   target-pair properties.
2. Closed-loop target source-retention and donor-transfer curves.
3. Flow-step by layer residual heatmaps with BH-significant cells marked.
4. Flow-step by tensor/action-group dimension heatmaps.
5. Destination curves for both phase-aligned physical-state blocks.
6. Token-position heatmaps after the 15-state token sweep is authorized and
   complete.
7. Paired pi0 versus pi0.5 commitment and mediation differences.

## Explicit nonclaims

- Commitment under a natural paired counterfactual is not robustness to an
  arbitrary perturbation.
- Activation patching does not establish a human-like symbolic plan.
- Different rollout phases cannot be ordered into a universal semantic
  hierarchy without a phase-matched design.
- A single eligible noise seed is not a replicated gripper result.
- Probe accuracy or clean decodability is not causal commitment.
- The current pi0.5 result does not determine whether pi0 or pi0.7 has the same
  timing; pi0 requires the matched control run, and public pi0.7 internal weights
  are unavailable.
