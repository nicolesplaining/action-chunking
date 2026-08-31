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
- [ ] Validate deterministic paired episodes and activation capture.
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
