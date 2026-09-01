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

The first practical pilot is positive but not confirmatory. Emitting the five
controls executed before the next replan after seven rather than ten velocity
evaluations preserved correct target-first contact in 15/15 eligible scene
clusters and the target-first-plus-eventual-success composite in 14/15, with
exactly 30% fewer evaluations and 30.0% median paired integration-latency
savings. A frozen confirmation with 500 episode pairs (500 episodes per
condition; 1,000 condition rollouts total) is pending.
