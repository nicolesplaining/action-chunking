# Action commitment in vision-language-action models

This project asks when an action-chunking VLA becomes causally committed to
properties of its plan. The primary system is the public `pi05_libero`
checkpoint; a LIBERO-fine-tuned pi0 model is the matched architectural control.

The study separates three questions that are often conflated:

1. **Formation:** when is a property visible in an intermediate clean-action
   estimate?
2. **Causal mediation:** which transformer layers and future action positions
   transmit the property?
3. **Commitment:** after which flow step does a natural-scale counterfactual
   conditioning switch fail to redirect the final plan?

The exact upstream OpenPI revision is pinned as a git submodule in
`third_party/openpi`. Experimental code extends that implementation without
copying or silently modifying it.

## Study status

- [x] Pin the official OpenPI implementation.
- [x] Specify the causal estimands, hypotheses, controls, and analysis rules.
- [ ] Reproduce the published pi0.5-LIBERO evaluation.
- [x] Validate deterministic paired episodes and activation capture.
- [ ] Run pilot interventions and lock confirmatory pair families.
- [ ] Run the confirmatory pi0.5 study.
- [ ] Fine-tune and evaluate the matched pi0 control.
- [ ] Produce the manuscript, figures, and reproducibility package.

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
  never described as causal commitment.

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
   refusing to assign commitment to properties with inadequate clean endpoint
   contrast.

Machine-specific checkpoint paths and hosts are CLI arguments and are never
stored in the repository.
