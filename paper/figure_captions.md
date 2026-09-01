# Figure captions

## Figure 1: Formation precedes loss of conditional editability

**a,** Mean relative error between each intermediate clean paired contrast and
the final clean contrast across 15 dual-success scene states; ribbons are 95%
scene-cluster bootstrap intervals and the dashed line is the preregistered 0.20
formation tolerance. **b,** Isotonic source retention under a suffix
conditioning switch. The horizontal line is the registered 0.80 editability
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

## Figure 3: Late behavioral editability at two rollout phases

**a,** Target-identity editability from task initialization across 15 scene
states and 30 directional curves. **b,** Post-grasp destination editability from
two independently selected held-object physical-state blocks and four
directional curves. Every curve is monotonic and both endpoint controls pass.
The destination reaches complete source retention after eight updates, while
target identity requires nine. These contrasts begin at different rollout
phases and are not interpreted as a universal semantic hierarchy. Ribbons are
95% scene-cluster bootstrap intervals; the line marks the registered 0.80
editability threshold.

## Figure 4: Late residual effects concentrate in deferred action positions

**a,** Full-action NCTE for the preregistered population position grid crossing
flow steps 0, 7, 8, and 9; action-expert layers 0, 8, 14, and 17; and all ten
future positions in 15 scene states. Open circles require both BH-adjusted
`q < 0.05` and a 95% scene-cluster interval above zero. The dotted line divides
the five positions executed before the next clean replan from positions 5--9,
which are deferred. **b,** Final-update, final-layer position profiles. The
preregistered executed-minus-deferred contrast is negative for full action,
translation, and rotation, with every interval excluding zero.

## Figure 5: Noise-conditional future-token localization of closure

Bidirectional closure-time NCTE for 120 single-position residual interchanges in
one eligible scene/noise mode. Only future action position 9 transfers closure
time. At flow step 6 the effect appears only after layer 17; at steps 7 and 8 it
appears at all four tested depth anchors. The paired closure contrast is absent
in the other three registered noise modes, so the all-seed eligible state count
is zero. This panel is descriptive and carries no population significance
markers.

## Figure 6: Seven evaluations preserve behavior in an exploratory pilot

**a,** Scene-cluster preservation under seven rather than ten velocity-field
evaluations at every replan. Correct target-first contact is preserved in all
15 eligible clusters; eventual dual-task success and their registered composite
are preserved in 14/15. The dashed line is the frozen descriptive pilot
threshold, not a noninferiority margin. **b,** Paired first-replan integration-
latency savings for the same 15 physical scene clusters. The line is the 30.0%
median and the band is its seed-zero 10,000-resample scene-bootstrap 95%
interval (27.4--32.9%). All differences favor early exit. The red cross marks
the one cluster that retained correct target-first contact but lost eventual
task success. The pilot reuses the mechanistic scene population; frozen paired
500-episode confirmation is pending.
