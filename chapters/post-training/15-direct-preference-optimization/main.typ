#import "../../../templates/typst/chapter.typ": technical-chapter, abstract, academic-table, chapter-bibliography

#show: technical-chapter.with(
  title: [Direct Preference Optimization],
)

#abstract[
  Direct Preference Optimization (DPO) trains an instruction-following policy directly from chosen--rejected response pairs. This chapter derives the DPO loss from a KL-regularized reward-maximization view of RLHF, showing how a policy and a fixed reference model define an implicit reward difference whose prompt-dependent normalization cancels in pairwise comparisons. It explains sequence-level log probabilities, the role of the coefficient $beta$, the offline training loop, and the practical distinction between DPO and Reward Model plus PPO pipelines. It also identifies the data, reference, and distributional limitations that a simpler objective does not remove.
]

= Introduction <sec-dpo-introduction>

Chapter 12 introduced Supervised Fine-Tuning (SFT), which imitates demonstrated instruction--response pairs. Chapter 13 then changed the supervision primitive from a demonstration to a comparison, and Chapter 14 showed how classical RLHF can train a Reward Model (RM) on those comparisons and optimize a policy with PPO. Direct Preference Optimization (DPO) takes a different route: it uses the preference pairs to update the policy directly, without first fitting a separately parameterized RM or running an on-policy actor--critic loop.

The two pipelines share their SFT starting point and their dependence on preference data:

$
  "SFT" arrow "Preference Data" arrow "Reward Model" arrow "RLHF + PPO",
  quad
  "SFT" arrow "Preference Data" arrow "DPO".
$ <eq-dpo-pipeline-comparison>

DPO is not ordinary SFT on the chosen response. Its loss compares how the trainable policy changes the relative likelihood of the chosen and rejected completions *against a fixed reference policy*. That comparison follows from a particular KL-regularized RLHF model under a Bradley--Terry preference assumption @rafailov2023dpo. The connection is useful, but it is not a claim that DPO and PPO expose the same operational interface: one consumes a fixed offline dataset, while the other repeatedly generates and scores new Rollouts.

= Preference Pairs and Sequence-Level Policy Scores <sec-dpo-preference-pairs>

Let $cal(P) = {(x_i, y_"w"^(i), y_"l"^(i))}_(i=1)^N$ be a dataset of prompts $x$, chosen responses $y_"w"$, and rejected responses $y_"l"$, using the meaning established in Chapter 13. Let $pi_theta$ be the trainable policy and $pi_"ref"$ be a frozen reference policy, commonly the SFT checkpoint that initializes $pi_theta$. Both policies must assign probabilities under exactly the same tokenizer, Chat Template, role markers, stopping convention, and response boundaries.

If $y = (y_0, dots, y_(T-1))$ is a response conditioned on $x$, its sequence-level log probability is the sum of teacher-forced token log probabilities,

$
  log pi_theta(y | x)
  = sum_(t=0)^(T-1)
    log pi_theta(y_t | x, y_0, dots, y_(t-1)).
$ <eq-dpo-sequence-log-probability>

Thus *token-level log probability* is one conditional term in @eq-dpo-sequence-log-probability, while *sequence-level log probability* is the log probability of the entire response. In a batched implementation, only assistant-response positions contribute to this sum. Prompt, padding, and positions after termination are context or invalid positions, not response targets. Whether the terminal token is included is a declared modeling choice, but the policy and reference must use the same choice.

The basic quantity in DPO is the log-probability change relative to the reference,

$
  q_theta(x, y)
  = log pi_theta(y | x) - log pi_"ref"(y | x).
$ <eq-dpo-reference-relative-score>

$q_theta(x, y)$ is not a reward observed from an annotator. It is a model-dependent likelihood ratio in log space: it is positive when the policy has made this response more likely than the reference and negative when it has made it less likely. Comparing the two responses gives the reference-relative policy margin

$
  Delta_theta(x, y_"w", y_"l")
  = q_theta(x, y_"w") - q_theta(x, y_"l").
$ <eq-dpo-policy-margin>

Increasing @eq-dpo-policy-margin makes the chosen completion relatively more probable and the rejected completion relatively less probable than they were under the reference. The subtraction is essential. A standalone high likelihood for $y_"w"$ is not enough: a policy can increase both responses, or favor the rejected response even more strongly.

= From KL-Regularized RLHF to an Implicit Reward <sec-dpo-derivation>

The RLHF objective in Chapter 14 conceptualizes a policy as maximizing a response-level reward while paying a KL cost for moving away from a reference. For one prompt $x$, write that objective as

$
  max_pi\; op("E")_(y ~ pi(. | x)) [
    r(x, y)
    - beta log frac(pi(y | x), pi_"ref"(y | x))
  ],
  quad beta > 0.
$ <eq-dpo-kl-regularized-objective>

Here $r(x,y)$ is a latent preference reward and $beta$ is the coefficient on reference-relative deviation. Under this objective, the optimal policy has the exponential-tilting form

$
  pi_r(y | x)
  = frac(
    pi_"ref"(y | x) exp(r(x,y) / beta),
    Z(x)
  ),
  quad
  Z(x) = sum_(y') pi_"ref"(y' | x) exp(r(x,y') / beta).
$ <eq-dpo-optimal-policy-form>

The normalizer $Z(x)$ sums over complete responses and is not a quantity that ordinary language-model training can enumerate. Taking logarithms and rearranging @eq-dpo-optimal-policy-form nevertheless yields

$
  r(x,y)
  = beta [
    log pi_r(y | x) - log pi_"ref"(y | x)
  ] + beta log Z(x).
$ <eq-dpo-implicit-reward-reparameterization>

The final term depends on the prompt but not on which candidate response is compared. The Bradley--Terry model from Chapter 13 uses only a reward difference. For two candidates, the prompt-only term in @eq-dpo-implicit-reward-reparameterization cancels:

$
  r(x,y_"w") - r(x,y_"l")
  = beta [
    q_r(x,y_"w") - q_r(x,y_"l")
  ].
$ <eq-dpo-reward-difference-cancellation>

This is the *implicit reward view*. If $pi_theta$ is used in place of the unknown optimal policy $pi_r$, then

$
  hat(r)_theta(x,y)
  = beta [log pi_theta(y | x) - log pi_"ref"(y | x)]
$ <eq-dpo-implicit-reward>

behaves as a reward parameterization inside the pairwise likelihood. It is not a separately trained scalar Reward Model that can independently score arbitrary responses. It is defined only through the policy and reference probabilities, and its scale depends on $beta$ and the chosen reference. The derivation establishes an equivalence under its modeling assumptions; it does not prove that the implicit quantity is a calibrated measure of human value @rafailov2023dpo.

== The Reference Model and $beta$ <sec-dpo-reference-beta>

The reference model has two roles. In the derivation, it defines the KL anchor in @eq-dpo-kl-regularized-objective. In the implemented loss, it makes DPO optimize a *relative* margin rather than merely rewarding frequent chosen strings. A response that the reference already strongly favors receives a different training signal from a response that becomes likely only after adaptation. The reference must remain frozen while a DPO run is interpreted; otherwise the target margin changes as the optimizer moves.

The coefficient $beta$ appears both as the KL coefficient in the derivation and as the scale on the preference margin in the resulting logistic loss. It therefore controls the sensitivity of the loss to a given policy-versus-reference change. Larger $beta$ makes a fixed relative margin more decisive in the sigmoid, and changing it changes the balance implied by the KL-regularized model. It should not be described as a universal measure of how strongly humans prefer one answer. Its useful range depends on model initialization, pair quality, response lengths, optimizer settings, and the desired amount of deviation from the reference.

= The DPO Objective <sec-dpo-objective>

Substituting @eq-dpo-reward-difference-cancellation into the Bradley--Terry likelihood gives the DPO loss

$
  cal(L)_"DPO"(theta)
  = - op("E")_((x,y_"w",y_"l") ~ cal(P)) [
    log sigma(
      beta [
        log pi_theta(y_"w" | x) - log pi_"ref"(y_"w" | x)
        - log pi_theta(y_"l" | x) + log pi_"ref"(y_"l" | x)
      ]
    )
  ].
$ <eq-dpo-objective>

Equivalently, the sigmoid argument is $beta Delta_theta$ from @eq-dpo-policy-margin. It is large and positive when the policy has increased the chosen response's log probability relative to the reference more than it has increased the rejected response's. The negative log-sigmoid then becomes small. If the policy favors the rejected response in relative terms, the margin is negative and the loss applies a larger correction. This is a binary classification loss over pair orderings, but its features are sequence likelihood ratios rather than an independently computed label embedding.

The gradient clarifies why DPO is more than maximizing the chosen likelihood. Let $z = beta Delta_theta$. Since

$
  frac(partial [- log sigma(z)], partial z) = sigma(z) - 1,
$ <eq-dpo-logistic-gradient>

the update is strongest for pairs whose current reference-relative ordering is uncertain or wrong. Differentiating $z$ changes $log pi_theta(y_"w" | x)$ upward and $log pi_theta(y_"l" | x)$ downward, with a coupling determined by the sigmoid. The reference log probabilities are constants in this differentiation. A naive unlikelihood-style objective that ignores the reference-relative weighting does not preserve this particular KL-regularized interpretation.

= Offline Training Loop and Its Trade-Offs <sec-dpo-training-loop>

The dataflow is substantially simpler than PPO, but it is still a careful likelihood computation rather than a format-agnostic label update.

#figure(
  block(width: 100%)[
    #set text(size: 9.05pt)
    #set par(justify: false, leading: 0.55em, spacing: 0pt)
    #academic-table(
      columns: (0.8fr, 1.55fr, 1.7fr),
      align: (center, left, left),
      inset: (x: 4pt, y: 2.7pt),
      header: (
        [*Stage*], [*Operation*], [*Invariant to preserve*],
      ),
      rows: (
        [1], [Load a preference pair], [Prompt, chosen and rejected response, label orientation, and data split remain recoverable.],
        [2], [Serialize two continuations], [Chat Template, tokenizer, response boundaries, terminal-token rule, and target masks match exactly.],
        [3], [Compute policy and reference log probabilities], [Both models score the same valid response-token positions; reference outputs are detached.],
        [4], [Form the DPO loss], [$beta$, sequence-level reductions, pair weighting, and valid-pair denominator are declared.],
        [5], [Update only the policy], [Optimizer state, gradient controls, checkpoints, and evaluation revisions are recorded.],
      ),
    )
  ],
  caption: [DPO is an offline pairwise likelihood loop. Each pair needs policy and reference likelihoods for both responses; it needs neither a separately trained RM, an actor--critic target, nor on-policy Rollouts.],
) <tab-dpo-training-loop>

The policy begins from SFT or a compatible instruction-following checkpoint. For each pair, the trainable policy and frozen reference score both completions by teacher forcing. The loss in @eq-dpo-objective is averaged over valid pairs, backpropagated through the policy only, and optimized with the ordinary training machinery discussed in Chapters 7 and 8. The reference can often be evaluated without gradient storage, but it remains a material memory and throughput cost unless an equivalent implementation strategy is used.

#figure(
  block(width: 100%)[
    #set text(size: 9.05pt)
    #set par(justify: false, leading: 0.55em, spacing: 0pt)
    #academic-table(
      columns: (1.25fr, 1.7fr, 1.7fr),
      align: (left, left, left),
      inset: (x: 4pt, y: 2.7pt),
      header: (
        [*Property*], [*DPO*], [*RM plus PPO RLHF*],
      ),
      rows: (
        [Primary supervision], [Fixed chosen--rejected preference pairs.], [Preference pairs train an RM; fresh Rollouts are then scored.],
        [Optimization signal], [Reference-relative sequence likelihood margin.], [RM reward, shaped returns, and estimated advantages.],
        [Learned auxiliary models], [No explicit RM or Value Model is required by the basic objective.], [An RM and normally a critic or value head are required.],
        [Policy data during update], [Offline dataset; no rollout refresh is inherent.], [On-policy responses from the current behavior policy.],
        [Principal strength], [Simple, stable supervised-style optimization interface.], [Can optimize newly sampled behavior and more flexible reward signals.],
        [Principal limitation], [Cannot correct missing coverage by searching for new feedback during a fixed run.], [More coupled models, sampling, variance, and stability controls.],
      ),
    )
  ],
  caption: [DPO and PPO-based RLHF use preference information differently. Neither table entry is an unconditional quality ranking: the preferred method depends on the feedback, reward interface, and need for online interaction.],
) <tab-dpo-versus-ppo>

@tab-dpo-versus-ppo identifies the practical transition. DPO avoids explicit RM fitting, rollout-based PPO optimization, Value Model training, and advantage estimation. It does not make preference learning free: it still needs correctly serialized pair data, policy and reference likelihoods for both responses, optimization controls, protected evaluation, and a defensible reference policy. PPO remains useful when a system can collect new on-policy behavior, needs a reward from a verifier, simulator, or other non-pairwise signal, or needs to optimize an objective whose feedback is not fully represented in a static comparison corpus. DPO is therefore a simpler offline preference-optimization pipeline, not a universally superior replacement for online policy optimization.

= Data Limits, Failure Modes, and Extensions <sec-dpo-failure-modes>

The DPO objective can faithfully optimize the information in a pair while still moving toward an undesirable policy. Incorrect label orientation, duplicated prompts, ambiguous or inconsistent annotations, low-quality synthetic judges, and candidates with systematic formatting artifacts all change the learned margin. The data-quality and annotation-noise analysis of Chapter 13 therefore remains a prerequisite rather than a stage DPO bypasses.

*Distribution shift* takes a different form from the RM-search problem in PPO. DPO does not deliberately sample new responses during a fixed offline update, so it does not use an RM to rank its own increasingly unusual Rollouts. It also cannot obtain preference supervision for behaviors absent from its dataset. If deployment prompts, response lengths, languages, or failure cases differ from the recorded pairs, the likelihood-ratio objective has no automatic corrective signal. A high held-out pair accuracy or a low DPO loss on an in-distribution split is not evidence of broad behavioral alignment.

Length deserves special attention. Standard DPO uses the sum in @eq-dpo-sequence-log-probability, so longer responses contain more token log-probability terms. Whether that creates a harmful preference depends on the pair distribution, response-length correlation, stop handling, and evaluation protocol; it cannot be diagnosed from the loss alone. Dividing by length changes the objective and should not be introduced as an innocent numerical stabilization. Independent length-controlled evaluation remains necessary, as it does for RMs and PPO policies.

The reference is another possible failure boundary. A weak or mismatched reference can make relative scores poorly aligned with the intended starting behavior. An overly aggressive optimization configuration can drift from the reference despite the derived interpretation, while a conservative configuration can make little meaningful change. The theoretical assumptions behind the derivation also include a specified preference model and adequate support of the reference policy. Real datasets contain finite, noisy, and often deterministically labeled comparisons; these conditions constrain what can be concluded from the formal equivalence. Analyses of direct preference objectives emphasize that DPO's offline formulation has its own regularization and overfitting behavior @gheshlaghi2024preference.

Several direct-preference methods alter the link function, regularization, target margin, or feedback setting. Identity Preference Optimization (IPO), for example, arises from a different choice within a broader preference-optimization formulation and is motivated in part by DPO's potential overfitting behavior under deterministic labels @gheshlaghi2024preference. Such variants are useful reminders that "direct preference optimization" names a family of design choices, not one interchangeable loss. This chapter does not develop GRPO, online preference optimization, or reasoning-specific objectives; those require their own data-collection and credit-assignment analysis.

= Implementation Contracts <sec-dpo-implementation-contracts>

The data contract must version the prompt and conversation serialization, tokenizer, Chat Template, role and special tokens, chosen/rejected orientation, annotation or judge source, preference rubric, candidate provenance, deduplication policy, split assignment, and response stop convention. It must reject or explicitly handle a pair with missing boundaries, an empty target span, unsupported roles, identical candidates, malformed terminal tokens, or an undecided label. A role reversal test should change the sign of @eq-dpo-policy-margin and convert the pair into its complementary training example.

The scoring contract must define the response-token mask used in @eq-dpo-sequence-log-probability, whether the end-of-sequence token belongs to the target, truncation and packing behavior, per-sequence reduction, and pair-level averaging. It must ensure that $pi_theta$ and $pi_"ref"$ score the identical token IDs at identical positions. Tests should verify that padding, prompt tokens, and post-termination positions contribute neither to the policy nor the reference response log probability; they should also verify that the reference path is frozen and detached from gradient updates.

The optimization contract must record the reference checkpoint, $beta$, policy initialization, precision configuration, optimizer and scheduler state, learning rate, effective batch size, gradient accumulation and clipping rules, pair weighting, checkpoint policy, and evaluation datasets. Logs should include chosen and rejected policy log probabilities, their reference-relative margin, DPO loss, response lengths, gradient statistics, finite-value checks, and protected task or human evaluation. Chapter 11's resumability requirements apply: recovering only the policy weights does not reproduce a run without its optimizer, scheduler, data-order, random-state, and configuration state.

= Summary <sec-dpo-summary>

DPO converts a chosen--rejected preference pair into a direct update of an autoregressive policy. The central objects are sequence-level log probabilities for the policy and a frozen reference. Their difference defines a reference-relative score; comparing that score for the chosen and rejected responses yields the margin optimized by the DPO logistic loss.

The formal bridge to RLHF begins with KL-regularized reward maximization. Its optimal policy can be written as an exponential tilt of the reference, allowing a reward difference to be represented by policy-to-reference log-probability differences. The prompt-only normalization cancels in the Bradley--Terry comparison, which is why an explicit RM need not be fitted for the basic DPO objective. The resulting method avoids rollout collection, PPO clipping, Value Model fitting, and advantage estimation during its offline update.

Those simplifications do not remove the limits of preference data. Annotation noise, response-length and style bias, reference dependence, distribution shift, and insufficient coverage remain policy-level risks. PPO-based RLHF and DPO therefore offer different interfaces to feedback: PPO can optimize new on-policy behavior and flexible rewards, whereas DPO offers a compact, offline likelihood objective for a fixed preference corpus. Choosing between them is a question about available feedback and evaluation evidence, not a generic ranking of algorithms.

#heading(level: 1, numbering: none, outlined: false)[References]
#chapter-bibliography("/references/handbook.bib")
