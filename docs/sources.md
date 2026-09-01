# Source provenance

The project uses public implementations and papers as the starting point.
Version-specific claims should cite the pinned source revision or an immutable
paper version, not this summary.

## Models and benchmarks

- **OpenPI:** Physical Intelligence's public pi0/pi0.5 implementation. Pinned at
  commit `215abfb217dbac7d5f1273282331b9b1866c0479` in
  `third_party/openpi`. The matched-pi0 checkpoint conversion reuses the pinned
  public converter's parameter mapping and adopts the narrowly scoped
  float32-intermediate safeguard proposed in public
  [OpenPI PR #978](https://github.com/Physical-Intelligence/openpi/pull/978).
  The original bfloat16 conversion failure is retained because public reports
  also document JAX/PyTorch output and performance discrepancies
  ([issue #810](https://github.com/Physical-Intelligence/openpi/issues/810),
  [issue #840](https://github.com/Physical-Intelligence/openpi/issues/840)).
- **pi0:** flow-matching VLA and action-expert architecture
  ([paper](https://arxiv.org/abs/2410.24164)).
- **pi0.5:** open-world VLA and heterogeneous co-training
  ([paper](https://arxiv.org/abs/2504.16054)). The public OpenPI release exposes
  the flow-matching action path but not the full high-level semantic-prediction
  training pipeline described in the paper.
- **pi0.7:** richer multimodal context conditioning
  ([paper](https://arxiv.org/abs/2604.15483)). It is a future extension because
  its trained implementation and weights are not present in the pinned OpenPI
  release. A fresh audit of the official
  [OpenPI model and checkpoint list](https://github.com/Physical-Intelligence/openpi)
  on 2026-08-31 likewise found public pi0, pi0-FAST, and pi0.5 entries but no
  pi0.7 implementation or checkpoint. Pi0.7 is therefore ineligible for internal
  activation interventions; a remote endpoint could only support a separately
  labeled behavioral comparison.
- **LIBERO:** lifelong robot-learning benchmark
  ([paper](https://arxiv.org/abs/2306.03310),
  [code](https://github.com/Lifelong-Robot-Learning/LIBERO)).
  Exact task-pair discovery reads the pinned public benchmark registry and BDDL
  files; it does not reconstruct task semantics from filenames.
- **robosuite 1.4.1:** LIBERO's public robot-control dependency
  ([paper](https://arxiv.org/abs/2009.12293),
  [code](https://github.com/ARISE-Initiative/robosuite/tree/v1.4.1)). The
  transition-exact recovery fork follows its public operational-space controller
  semantics and reconstructs controller/interpolator state by action replay;
  MuJoCo flat-state equality alone is not treated as transition equality.

## Action chunking and flow policies

- **ACT:** action chunking with transformer action queries
  ([paper](https://arxiv.org/abs/2304.13705),
  [code](https://github.com/tonyzhaozh/act)).
- **Diffusion Policy:** conditional denoising of action sequences and
  receding-horizon execution
  ([paper](https://arxiv.org/abs/2303.04137),
  [code](https://github.com/real-stanford/diffusion_policy)).

## Efficient and constrained generative control

- **Streaming Flow Policy:** changes the learned flow so successive integration
  states are executable robot actions, enabling online streaming rather than
  constructing and discarding intermediate action trajectories
  ([paper](https://proceedings.mlr.press/v305/jiang25a.html),
  [code](https://github.com/siddancha/streaming-flow-policy)). It is the closest
  public precedent for acting before a conventional action-flow sampler has
  fully completed. Our early-exit experiment instead leaves released pi0.5
  weights and flow semantics unchanged and emits the current predicted clean
  endpoint, so it does not inherit Streaming Flow Policy's training-time
  guarantees.
- **One-Step Diffusion Policy:** distills an iterative visuomotor diffusion
  policy to one action-generation step and evaluates success together with
  action-prediction frequency
  ([paper](https://arxiv.org/abs/2410.21257),
  [project](https://research.nvidia.com/labs/cosmos-lab/onedp/)). Its explicit
  neural-function-evaluation and latency reporting motivate our paired
  post-update compute accounting. We do not reuse its distillation objective
  because the present experiment tests training-free continuation of a fixed
  public VLA.
- **One-Step Flow Policy:** self-distills flow policies and includes a pi0.5
  integration, providing a direct contemporary reference for the practical cost
  of iterative VLA sampling ([paper](https://arxiv.org/abs/2603.12480)). Our
  continuation method is complementary: it saves only already-completed updates
  after a late instruction change and requires no retraining.
- **SafeDiffuser:** inserts control-barrier constraints into iterative diffusion
  planning and evaluates safety jointly with planning quality
  ([paper](https://arxiv.org/abs/2306.00148),
  [code](https://github.com/Weixy21/SafeDiffuser)). We use its separation of
  constraint satisfaction, task performance, and per-step overhead to structure
  the later obstacle/safety correction experiment; we do not claim its formal
  barrier guarantees for OpenPI.

## Inference-time steering and warm starts

- **DynaGuide:** steers a frozen diffusion policy during denoising with gradients
  from a separately trained latent dynamics model
  ([paper](https://arxiv.org/abs/2506.13922),
  [code](https://github.com/MaxDu17/DynaGuide)). It is the closest public
  precedent for changing an action sample during iterative generation. Our
  conditioning switch requires neither a learned dynamics model nor an
  auxiliary objective, but it can only express goals already understood by the
  VLA.
- **DSRL:** adapts diffusion and flow policies through reinforcement learning in
  latent-noise space and includes a public pi0 implementation
  ([paper](https://arxiv.org/abs/2506.15799),
  [code](https://github.com/nakamotoo/dsrl_pi0)). It establishes a strong pi0
  steering baseline, while our experiment changes the instruction condition of
  a fixed sample and makes no policy update.
- **Guided Action Flow:** applies learned Q-guidance to a frozen flow-matching
  VLA during sampling and evaluates on LIBERO
  ([paper](https://arxiv.org/abs/2607.02092),
  [code](https://github.com/ylhaichen/guided-action-flow)). Its critic-guided
  sampler is a natural future baseline for the obstacle/safety extension. The
  present target-retargeting test intentionally asks first whether the VLA's own
  instruction pathway can provide a useful correction without a critic.
- **STEP:** warm-starts diffusion-policy inference with a learned
  spatiotemporal consistency predictor and refines the proposal with a small
  number of denoising steps
  ([paper](https://arxiv.org/abs/2602.08245),
  [code](https://github.com/Kimho666/STEP)). It motivates treating continuation
  quality and saved neural-function evaluations as a joint Pareto question, not
  interpreting fewer updates as useful without closed-loop success.
- **Consistency Policy:** distills a diffusion policy into a single- or
  few-step policy for low-latency visuomotor inference
  ([paper](https://arxiv.org/abs/2405.07503),
  [code](https://github.com/Aaditya-Prasad/Consistency-Policy)). Distillation is
  complementary to our training-free reuse of updates already computed before
  an instruction changes.

## Causal methodology

- Interchange interventions and causal abstraction: Geiger et al.
  ([paper](https://arxiv.org/abs/2112.00826)).
- Activation-patching metrics and sensitivity to methodological choices: Zhang
  and Nanda ([paper](https://arxiv.org/abs/2309.16042)).
- General causal-abstraction framework: Geiger et al.
  ([paper](https://arxiv.org/abs/2301.04709)).

## VLA interpretability

- **Mechanistic Interpretability for Steering VLAs:** FFN value-vector
  interpretation and causal activation steering in OpenVLA and pi0/FAST
  ([paper](https://openreview.net/forum?id=YvsUD8C9QS),
  [code](https://github.com/Physical-AI-Safety-Institute/mechanistic-steering-vlas)).
  It establishes that internal interventions can causally steer VLA behavior;
  our study asks the distinct question of *when* natural paired conditioning
  loses causal control across action-expert layers and flow integration.
- **DR.VLA:** sparse-autoencoder features in pi0.5 PaliGemma and action-expert
  activations, including per-token analyses and causal steering
  ([paper](https://arxiv.org/abs/2603.19183),
  [project/code](https://drvla.github.io/)). Its distinction between general
  features and episode memorization motivates cross-scene replication rather
  than interpreting a single high-effect site.
- **Event-SAE:** event-grounded feature ranking and residual-preserving
  interventions for OpenVLA and pi0.5
  ([paper](https://arxiv.org/abs/2605.17204),
  [code](https://github.com/xc-j/Event-SAE)). Its public kinematic-event and
  keyframe pipeline is the preferred starting point for grasp, transport, and
  recovery phase labels; any reused module will be pinned and attribution kept.
  The current rollout logger emits its public `trajectory_records.jsonl` field
  schema directly. The implementation audited for this adapter is commit
  `f7a000024a32d8b9ee8e92aab5e79694a2f2bc1c`; the AWE threshold will be
  calibrated on pilot trajectories rather than copied without validation. The
  corresponding public AWE fork is pinned at
  `7197bb86a20784666dabed90e6eabcf8bb1e9912`; the versioned launcher verifies
  both revisions before calling Event-SAE's extractor unchanged.

These works primarily interpret or steer features at selected layers or rollout
times. None of them replaces the paired flow-switch experiment here, which
crosses transformer depth, future-token position, and every flow step while
holding images, state, and initial action noise fixed.

## Source-use policy

1. Reuse upstream preprocessing, checkpoint loading, model definitions, and
   benchmark adapters when available.
2. Pin every external source revision and document any local behavioral change.
3. Keep analysis and intervention code in this repository; do not fork model
   code unless hooks cannot express the required intervention.
4. Validate inferred architecture details against both the paper and executable
   source. When they differ, the executable checkpoint path governs the
   experiment and the discrepancy is reported.
5. Before implementing a new phase detector, activation collector, or rollout
   evaluator, audit the three public VLA-interpretability codebases above and
   adapt a pinned implementation when its semantics match the preregistered
   outcome.
