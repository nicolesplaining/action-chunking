# Figure captions

## Figure 1: Formation is not commitment

**a,** Mean relative error between each intermediate clean paired contrast and
the final clean contrast across 15 dual-success scene states; ribbons are 95%
scene-cluster bootstrap intervals and the dashed line is the preregistered 0.20
formation tolerance. **b,** Isotonic source retention under a suffix
conditioning switch. The horizontal line is the registered 0.80 commitment
threshold. Full action, translation, and target direction do not cross it until
the final update; rotation crosses at update nine. **c,** Repeated closed-loop
target-identity intervention. Source retention remains zero through seven
source-conditioned updates, rises to 0.60 after eight, and reaches one after
nine. Ribbons are 95% scene-cluster bootstrap intervals over 15 physical scene
states; both intervention directions are included.

## Figure 2: Causal localization across depth and action state

**a,** Mean bidirectional normalized causal transfer (NCTE) from patching all
future action-token residuals at each action-expert layer and flow step. Red is
directed donor transfer and blue is counter-directed transfer. Open circles mark
positive cells with both Benjamini--Hochberg-adjusted `q < 0.05` and a 95%
scene-cluster interval strictly above zero. The largest positive cell for all
three outcomes is layer 17 at flow step 9 (NCTE 0.095--0.099). **b,**
Property-matched coordinate interchanges in the evolving action state `x_t`
(solid) and predicted velocity `v_t` (dashed). Final-step `x_t` swaps transfer
0.537 of translation and 0.900 of rotation. Ribbons are 95% scene-cluster
bootstrap intervals over 15 states.

## Figure 3: Late behavioral commitment at two rollout phases

**a,** Target-identity commitment from task initialization across 15 scene
states and 30 directional curves. **b,** Post-grasp destination commitment from
two independently selected held-object physical-state blocks and four
directional curves. Every curve is monotonic and both endpoint controls pass.
The destination reaches complete source retention after eight updates, while
target identity requires nine. These contrasts begin at different rollout
phases and are not interpreted as a universal semantic hierarchy. Ribbons are
95% scene-cluster bootstrap intervals; the line marks the registered 0.80
commitment threshold.

## Figure 4: Noise-conditional future-token localization of closure

Bidirectional closure-time NCTE for 120 single-position residual interchanges in
one eligible scene/noise mode. Only future action position 9 transfers closure
time. At flow step 6 the effect appears only after layer 17; at steps 7 and 8 it
appears at all four tested depth anchors. The paired closure contrast is absent
in the other three registered noise modes, so the all-seed eligible state count
is zero. This panel is descriptive and carries no population significance
markers.
