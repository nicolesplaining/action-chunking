# Conditional editability in vision-language-action models

This project asks when a flow-matching action sample can still be causally
redirected, and whether that boundary is useful for low-latency retargeting.
The primary system is the public `pi05_libero` checkpoint; a LIBERO-fine-tuned
pi0 model is the matched architectural control.

The study separates three questions that are often conflated:

1. **Formation:** when is a property visible in an intermediate clean-action
   estimate?
2. **Causal mediation:** which transformer layers and future action positions
   transmit the property?
3. **Conditional editability:** after which flow step does a natural-scale
   counterfactual conditioning switch fail to redirect the sampled action?

Conditional editability is not a claim that the policy was previously
undecided or lacked an internal plan. A fully formed plan can remain
overwritable, and late redirection can reflect strong instruction following.
The registered utility study therefore asks whether the measured boundary
predicts successful mid-sampling retargeting, subsequent clean replanning, and
post-update latency.

The exact upstream OpenPI revision is pinned as a git submodule in
`third_party/openpi`. Experimental code extends that implementation without
copying or silently modifying it.

## Study status

- [x] Pin the official OpenPI implementation.
- [x] Specify the causal estimands, hypotheses, controls, and analysis rules.
- [x] Reproduce the published pi0.5-LIBERO evaluation.
- [x] Validate deterministic paired episodes and activation capture.
- [ ] Run pilot interventions and lock confirmatory pair families.
- [ ] Run the confirmatory pi0.5 study.
- [ ] Fine-tune and evaluate the matched pi0 control.
- [ ] Produce the manuscript, figures, and reproducibility package.

Current pilot evidence includes a 15-state prompt-only target contrast whose
clean action difference is formed near the start of denoising but remains
causally editable until the last one or two flow updates; a two-state-block
destination intervention whose four directional curves all switch between flow
boundaries 7 and 8; and a clean target-pose gripper contrast whose categorical
closure time loses editability at boundary 7. A one-state dynamic-retargeting
pilot also matched restart success at boundary 7 with 70% fewer post-change
velocity evaluations and about half the measured latency. However, boundaries
8--10 also succeeded because subsequent clean replanning recovered from a benign
first chunk, so this pilot does not show that editability timing predicts the
last successful recovery point. These are pilot results, not confirmatory
population claims. Exact controls, exclusions, intervals, and negative pilots
are recorded in `docs/reproducibility_log.md`.

The preregistered protocol is in
[`docs/research_protocol.md`](docs/research_protocol.md). Source provenance is
recorded in [`docs/sources.md`](docs/sources.md) and
[`references/references.bib`](references/references.bib).

## Reproducibility principles

- Paired runs share simulator state, robot state, preprocessing, and Gaussian
  action noise; exactly one named causal variable changes.
- The unit of statistical inference is the scene pair, not an individual noise
  seed or transformer token.
- Pilot pairs are excluded from confirmatory estimates.
- Interventions include identity, unrelated-donor, full-donor, direction-swap,
  and behavioral-validity controls.
- Results retain the model checkpoint hash, OpenPI commit, configuration,
  environment lock, seeds, and raw intervention manifest.
- Claims about representation use causal interventions; probe accuracy alone is
  never described as loss of conditional editability.

## Upstream implementation

Initialize the pinned source with:

```bash
git submodule update --init --recursive
```

OpenPI has its own installation and LIBERO instructions under
`third_party/openpi`. This repository will keep model-specific patches in a
small, reviewable adapter rather than editing upstream files.

## Experiment pipeline

The staged tools deliberately keep simulator generation, clean screening,
intervention, and analysis separate so patched outcomes cannot leak into pair
selection.

1. `catalog_libero_instruction_pairs.py` audits public BDDL files for strict
   single-variable task pairs. `generate_libero_instruction_pairs.py` and
   `run_instruction_pair_generation.sh` then serialize prompt-only fixtures and
   enforce byte-exact equality of images, proprioception, simulator state, and
   object poses. `generate_instruction_pair_grid.sh` retains the original
   six-contrast LIBERO-90 pilot grid.
   `generate_postgrasp_instruction_pairs.py` constructs phase-aligned
   destination fixtures, while `generate_target_pose_pairs.py` changes only a
   target free joint's registered planar coordinates for closure-timing pairs.
2. `screen_instruction_pairs.py` evaluates only clean chunks under shared saved
   noise and writes a pre-intervention eligibility table.
3. `run_pair_interventions.py` runs bidirectional flow switches, residual-stream
   patches, future-token patches, and grouped/scalar `x_t` or `v_t` dimension
   interchanges. Every run retains raw actions and exact controls.
4. `serve_noise_policy.py`, `run_pair_validation.sh`, and
   `validate_libero_pair_rollouts.py` replay the same noise sequence in both
   closed-loop task environments and verify the first chunks against clean-only
   offline inference.
5. `analyze_pair.py` emits machine-readable tables and pilot figures while
   refusing to assign an editability boundary to properties with inadequate
   clean endpoint contrast.
6. `make_manuscript_figures.py` composes the frozen population tables into four
   compact PDF/PNG figures and writes a manifest containing every input SHA-256.
   Population heatmap markers require both BH-adjusted `q < 0.05` and a
   scene-cluster interval strictly above zero; the noise-conditional closure
   panel is explicitly labeled descriptive.

Machine-specific checkpoint paths and hosts are CLI arguments and are never
stored in the repository.
