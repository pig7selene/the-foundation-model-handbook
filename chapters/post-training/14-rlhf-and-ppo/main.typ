#import "../../../templates/typst/chapter.typ": technical-chapter, abstract, academic-table, chapter-bibliography

#show: technical-chapter.with(
  title: [RLHF and PPO],
)

#abstract[
  Reinforcement Learning from Human Feedback (RLHF) uses a learned preference signal to improve the behavior of an instruction-following language model beyond direct imitation. This chapter formulates an autoregressive language model as a token-level policy, derives the policy-gradient estimator used to optimize sequence-level rewards, and introduces the actor-critic and Generalized Advantage Estimation machinery that reduces its variance. It then develops Proximal Policy Optimization, distinguishes its clipped surrogate from the separate reference-model KL constraint used in language-model RLHF, and closes with the practical PPO loop, common failure modes, and implementation contracts.
]

= Introduction <sec-rlhf-introduction>

Chapter 12 established an SFT policy that imitates instruction-response demonstrations. Chapter 13 then showed how pairwise comparisons can train a Reward Model (RM) to rank complete responses. The next step is not another static supervised dataset update. It is an optimization problem in which a policy generates new responses, receives a score from the RM, and changes its next-token probabilities to increase expected reward while remaining close to useful SFT behavior.

The classical sequence is

$
  "Pretraining" arrow "SFT" arrow "Reward Model"
  arrow "RLHF" arrow "PPO".
$ <eq-rlhf-pipeline>

In this chapter, *RLHF* denotes the policy-optimization stage that consumes the SFT initialization, a fixed or slowly refreshed RM, a prompt distribution, and a reference policy. *PPO* is the policy-gradient method used to make that optimization practical. The discussion is deliberately narrow: it supplies the reinforcement-learning notation needed for language-model post-training, but does not attempt a general reinforcement-learning textbook or introduce DPO and GRPO in depth.

= Autoregressive Language Models as Policies <sec-rlhf-policy-model>

Let $x$ be a prompt sampled from a declared prompt distribution $cal(D)_"prompt"$. A generated response is $y = (a_0, dots, a_(T-1))$, where each $a_t in cal(V)$ is a token. At generation step $t$, define the state as the prompt together with the generated prefix,

$
  s_t = (x, a_0, dots, a_(t-1)),
  quad
  a_t ~ pi_theta(. | s_t).
$ <eq-llm-policy-state-action>

The policy $pi_theta$ is the language model's next-token distribution. The environment transition is almost deterministic: appending $a_t$ produces $s_(t+1)$, except for stopping rules, truncation, and any external tool or simulator that a later application may introduce. A complete response is a trajectory, and its probability factorizes as

$
  pi_theta(y | x) = product_(t=0)^(T-1) pi_theta(a_t | s_t).
$ <eq-llm-trajectory-probability>

In the common outcome-reward setting, the RM produces one scalar after a response is complete: $r_psi(x, y)$. The model still takes one token action at a time, but the central quality signal is assigned at the end of the trajectory. This delayed, sampled reward is the source of both the flexibility and the difficulty of RLHF.

#figure(
  block(width: 100%)[
    #set text(size: 9.05pt)
    #set par(justify: false, leading: 0.55em, spacing: 0pt)
    #academic-table(
      columns: (1.2fr, 1.65fr, 1.65fr),
      align: (left, left, left),
      inset: (x: 4pt, y: 2.7pt),
      header: (
        [*RL object*], [*LLM realization*], [*Role in the PPO stage*],
      ),
      rows: (
        [State $s_t$], [Prompt plus all tokens generated before position $t$.], [Conditions the next-token distribution and value estimate.],
        [Action $a_t$], [One sampled vocabulary token.], [Receives a policy-gradient contribution through its log probability.],
        [Trajectory $y$], [A terminated or truncated assistant response.], [Is scored by the RM and supplies token-level returns.],
        [Policy $pi_theta$], [The trainable autoregressive language model.], [Is improved from fresh rollouts using the clipped surrogate.],
        [Reference $pi_"ref"$], [A fixed SFT checkpoint, commonly token-compatible with the policy.], [Defines the KL anchor, not the rollout behavior policy.],
      ),
    )
  ],
  caption: [The RL abstraction preserves the autoregressive structure of a language model. The same completed response is a token trajectory for the policy and a sequence-level object for the Reward Model.],
) <tab-llm-rl-objects>

@tab-llm-rl-objects exposes a crucial distinction. The RM is a function that scores a completed $(x, y)$ pair. The policy is a conditional distribution that must choose every token along $y$. Backpropagating through the RM does not directly differentiate through discrete samples $a_t$ or through the combinatorial set of possible continuations. Policy-gradient estimation supplies a way to improve the distribution from sampled trajectories instead.

= Objective, Reward Shaping, and Policy Gradients <sec-rlhf-objective>

Let the unregularized objective be the expected RM score,

$
  J_"RM"(theta)
  = op("E")_(x ~ cal(D)_"prompt", y ~ pi_theta(. | x))
    [r_psi(x, y)].
$ <eq-rm-expected-return>

Optimizing @eq-rm-expected-return directly is difficult for three linked reasons. The response distribution changes when $theta$ changes, a response contains many sampled discrete decisions, and RM scores are only approximations to the judgment one ultimately wants. A model that changes its distribution aggressively may visit responses on which the RM was never reliable. Classical language-model RLHF consequently regularizes the policy toward a fixed reference model, often the SFT checkpoint.

For each state, define the forward token-distribution divergence

$
  op("KL")_t(theta)
  = sum_(a in cal(V)) pi_theta(a | s_t)
    log frac(pi_theta(a | s_t), pi_"ref"(a | s_t)).
$ <eq-token-kl-divergence>

One conceptual regularized objective is

$
  J(theta)
  = op("E")_(x ~ cal(D)_"prompt", y ~ pi_theta(. | x))[
      r_psi(x, y) - beta sum_(t=0)^(T-1) op("KL")_t(theta)
    ],
  quad beta >= 0.
$ <eq-kl-regularized-rlhf-objective>

The coefficient $beta$ controls the trade-off. A small value permits greater reward-driven deviation; a large value favors staying near the reference and can make learning nearly inert. The expected KL term is nonnegative, though a sampled per-token log-ratio estimate can be positive or negative. For a trajectory sampled by the behavior-policy snapshot $pi_"old"$, many implementations use a sampled estimate to construct shaped rewards,

$
  tilde(r)_t^"old"
  = 1_(t = T-1) r_psi(x, y)
    - beta log frac(pi_"old"(a_t | s_t), pi_"ref"(a_t | s_t)),
$ <eq-token-reward-shaping>

where the RM score is placed at the final response position and the KL cost is distributed across generated tokens. At collection time, $pi_"old"$ is the current policy snapshot, so @eq-token-reward-shaping estimates the objective in @eq-kl-regularized-rlhf-objective. Once stored, these rewards, returns, and advantages are held fixed during PPO epochs; the changing $pi_theta$ then appears through the importance ratio. This changes credit assignment, not the intended trade-off: every token is charged for deviation from the reference, while the final response obtains the outcome score. Fine-Tuning Language Models from Human Preferences and InstructGPT are foundational examples of language-model reward optimization with a reference-model constraint @ziegler2019finetuning @ouyang2022training.

The log-derivative identity gives a policy-gradient estimator. Let $G_t^"old" = sum_(k=t)^(T-1) gamma^(k-t) tilde(r)_k^"old"$ be the return from position $t$, with discount $gamma in [0, 1]$. At rollout collection, the estimator is

$
  nabla_theta J(theta)
  = op("E")_[
      sum_(t=0)^(T-1)
      nabla_theta log pi_theta(a_t | s_t) G_t^"old"
    ].
$ <eq-policy-gradient-return>

@eq-policy-gradient-return says that actions preceding a high return should become more likely, while actions preceding a low return should become less likely. It does not say that the last token alone caused the RM score. Every earlier token receives a shared, high-variance learning signal through its effect on the sampled continuation. The policy-gradient theorem formalizes this score-function route for parameterized policies @sutton2000policy.

== Return, Value, and Advantage <sec-rlhf-returns-advantages>

The four quantities below serve different purposes. A *reward* $tilde(r)_t^"old"$ is the immediate scalar assigned at one transition. A *return* $G_t^"old"$ is the discounted sum of future rewards. The state value is the expected future return under a policy,

$
  V^pi(s_t) = op("E")_[G_t^"old" | s_t].
$ <eq-state-value-definition>

The action value $Q^pi(s_t, a_t)$ is the expected return after taking $a_t$ and following $pi$ thereafter. The *advantage* is their difference,

$
  A^pi(s_t, a_t) = Q^pi(s_t, a_t) - V^pi(s_t).
$ <eq-advantage-definition>

An advantage asks whether this action was better or worse than the policy's expected continuation from the same state. Subtracting a state-only baseline such as $V^pi(s_t)$ leaves the expected policy gradient unchanged while greatly reducing variance. Thus reward is the supplied signal, return is accumulated reward, value is a prediction of expected return, and advantage is a centered action-quality estimate. Treating them as interchangeable hides the function of the critic.

= Actor-Critic Estimation and GAE <sec-rlhf-actor-critic>

An actor-critic implementation learns both a policy and a value predictor. The *actor* is $pi_theta$, which samples and assigns log probabilities to tokens. The *critic* is $V_phi(s_t)$, which predicts the shaped future return; it may use a separate model or a value head on a shared backbone. The critic is not the Reward Model. The RM maps a completed response to a preference-derived score and is normally frozen during a PPO update. The critic predicts the future reward that the current rollout process will produce and is trained from rollout returns.

For a sampled trajectory, define the temporal-difference residual

$
  delta_t
  = tilde(r)_t^"old" + gamma V_phi(s_(t+1)) - V_phi(s_t),
$ <eq-td-residual>

with $V_phi(s_T) = 0$ after termination. Generalized Advantage Estimation (GAE) forms

$
  hat(A)_t^"GAE"
  = sum_(l=0)^(T-t-1) (gamma lambda)^l delta_(t+l),
  quad lambda in [0, 1].
$ <eq-generalized-advantage-estimation>

When $lambda$ is near one, @eq-generalized-advantage-estimation retains longer-horizon information and generally has less bootstrap bias but more variance. When it is near zero, it relies more heavily on the critic and has lower variance but more bias if the value estimate is inaccurate. Many language-model formulations use $gamma = 1$ because a finite response is not naturally time-discounted; $lambda$ then still controls the bias-variance trade-off in the advantage estimator. GAE was introduced as a practical variance-reduction estimator for policy gradients @schulman2015gae.

The value function is commonly trained by regression to a declared target $hat(G)_t$, for example a bootstrapped return compatible with the chosen GAE convention:

$
  cal(L)_"value"(phi)
  = op("E")_t[(V_phi(s_t) - hat(G)_t)^2].
$ <eq-value-function-loss>

Policy loss and value loss therefore have different targets. The policy loss changes action probabilities according to estimated advantage. The value loss teaches a predictor of shaped future return. A low value loss does not prove that the policy is good; a high RM score does not prove that the critic is accurate.

= From Importance Sampling to PPO <sec-rlhf-ppo>

Rollouts are collected from a behavior policy $pi_"old"$, a snapshot of the policy before the current optimization epoch. Once those samples have been stored, the probability ratio for their token actions is

$
  r_t(theta)
  = frac(pi_theta(a_t | s_t), pi_"old"(a_t | s_t)).
$ <eq-importance-ratio>

Importance sampling uses @eq-importance-ratio to estimate how the new policy would weight actions sampled from the old policy. A simple surrogate is $op("E")[r_t(theta) hat(A)_t]$. It enables more than one minibatch pass over a rollout batch, but it becomes unreliable when the new policy moves far from the behavior policy: rare actions can receive large ratios, and a noisy advantage can drive a destructive update.

Proximal Policy Optimization replaces this surrogate with the clipped objective

$
  cal(L)^"CLIP"(theta)
  = op("E")_[
      op("min")(
        r_t(theta) hat(A)_t,
        op("clip")(r_t(theta), 1 - epsilon, 1 + epsilon) hat(A)_t
      )
    ].
$ <eq-ppo-clipped-objective>

For a positive advantage, increasing the probability ratio is useful only until it exceeds $1 + epsilon$; the clipped branch then prevents additional objective gain. For a negative advantage, decreasing the ratio is useful only down to $1 - epsilon$. The minimum in @eq-ppo-clipped-objective makes the surrogate pessimistic whenever the ratio leaves this interval in the direction that would otherwise improve the objective. PPO therefore constrains the incentive for one batch of noisy rollout data to induce a very large policy step, while still permitting several minibatch updates. PPO was introduced as a practical clipped surrogate for policy-gradient optimization @schulman2017ppo.

== PPO Clipping and Reference KL Are Not the Same <sec-rlhf-ppo-kl-distinction>

PPO clipping compares $pi_theta$ with $pi_"old"$, the policy that generated the current on-policy rollouts. Its role is local: stabilize reuse of one rollout batch while the policy is updated. Reference KL compares $pi_theta$ with $pi_"ref"$, the fixed SFT policy. Its role is global: preserve a behavioral anchor while the RLHF process moves across many rollout-and-update cycles.

Both mechanisms can limit drift, but neither replaces the other. A clipped update can still make steady, harmful movement away from SFT over many batches. A KL penalty alone can still permit an unstable optimization step relative to $pi_"old"$. In language-model RLHF, the reference constraint also addresses a special problem: the RM is a learned proxy whose errors become easier to exploit as the policy searches beyond its training distribution. Chapter 13's reward-overoptimization warning remains active throughout PPO.

= The PPO Loop for Language Models <sec-rlhf-ppo-loop>

One RLHF iteration begins by drawing a batch of prompts, not by reusing a static preference pair as though it were an on-policy trajectory. The current policy generates Rollouts under fixed decoding and stopping rules. The RM scores each completed response, the reference model supplies token log probabilities, and the critic supplies values. The system constructs shaped rewards, returns, and advantages, then performs a limited number of PPO minibatch updates before discarding the rollout batch and sampling again.

#figure(
  block(width: 100%)[
    #set text(size: 9.05pt)
    #set par(justify: false, leading: 0.55em, spacing: 0pt)
    #academic-table(
      columns: (0.8fr, 1.25fr, 2.0fr),
      align: (center, left, left),
      inset: (x: 4pt, y: 2.6pt),
      header: (
        [*Stage*], [*Operation*], [*State that must remain attached*],
      ),
      rows: (
        [1], [Sample prompts], [Prompt distribution revision, Chat Template, tokenizer, and prompt identifiers.],
        [2], [Generate Rollouts], [Behavior-policy revision, token IDs, stop reason, sampled log probabilities, and decoding settings.],
        [3], [Score and shape reward], [RM revision, reference log probabilities, KL coefficient, terminal-score convention, and reward normalization.],
        [4], [Estimate critic targets], [Value predictions, masks, returns, advantages, and valid-token reductions.],
        [5], [Run PPO epochs], [Old-policy log probabilities, clip range, policy and value losses, optimizer state, and minibatch schedule.],
        [6], [Evaluate and repeat], [External evaluation, RM-score diagnostics, KL statistics, response length, and checkpoint identity.],
      ),
    )
  ],
  caption: [A PPO iteration is an on-policy dataflow, not supervised training on static preference pairs. Each stored rollout needs the model revisions and token-level quantities that defined its objective.],
) <tab-ppo-loop-contract>

The policy, RM, value model, and reference model have different update schedules. The policy and critic usually change during PPO. The reference normally remains fixed for a run or a defined phase. The RM may remain frozen so that the objective is stable, or be refreshed only through a separately versioned preference-data procedure. Mixing these revisions without recording them makes a reward curve uninterpretable.

== Stability, Sensitivity, and Reward Overoptimization <sec-rlhf-stability>

PPO reduces but does not eliminate sensitivity. A large policy learning rate, many epochs over one rollout batch, a broad clip range, or a poorly normalized advantage can cause unstable updates. An inaccurate critic makes $hat(A)_t^"GAE"$ noisy or biased; a small rollout batch makes its variance high. Batch size changes both the statistical quality of the advantage estimate and the number of tokens through which the RM score is attributed. Chapters 7 and 8 remain relevant: optimizer settings, gradient clipping, precision, and finite-value checks still control the update path.

KL behavior supplies a useful but incomplete diagnostic. Persistently near-zero KL can indicate an overly strong $beta$, a weak reward signal, or an update regime that is not moving the policy. Rapidly growing KL can indicate excessive drift, but a moderate aggregate KL does not prove that every behavior remains safe or useful. Adaptive KL control can target a declared range, yet it cannot compensate for an RM that ranks the wrong property.

Reward overoptimization is visible when RM reward rises while independent human or task-grounded evaluation stagnates or falls. Common mechanisms include response-length exploitation, polished but unresponsive verbosity, mode collapse toward a narrow high-scoring style, and exploitation of RM artifacts outside the preference-data distribution. The remedy is not simply more PPO clipping. It requires protected evaluation, diverse prompts and candidate distributions, RM audits, conservative policy updates, and explicit decisions about the desired trade-off. The language-model preference-learning literature documents that optimizing a learned reward can exploit labeler heuristics and proxy errors @ziegler2019finetuning @eisenstein2023helping.

= Implementation Contracts <sec-rlhf-implementation-contracts>

The rollout contract must version the prompt distribution, Chat Template, tokenizer, current policy checkpoint, sampling parameters, maximum response length, stop-token policy, and response parsing rule. It must retain token IDs, attention masks, action masks, sampled old-policy log probabilities, and the terminal condition. A prompt token is context, not an RL action; padding and post-termination positions must contribute to neither policy loss nor value loss.

The reward contract must name the RM revision, its sequence serialization and scalar-readout convention, the reference revision, the exact KL estimator, the coefficient $beta$, reward normalization policy, and the terminal reward placement in @eq-token-reward-shaping. Tests should confirm that reference log probabilities and RM scores are evaluated on exactly the response tokens generated by the behavior policy. A changed Chat Template, stop convention, or response boundary changes the reward function operationally.

The optimization contract must distinguish $pi_"old"$ from $pi_"ref"$, preserve the old log probabilities used in @eq-importance-ratio, and record $epsilon$, $gamma$, $lambda$, PPO epoch count, minibatch construction, policy learning rate, value-loss coefficient, gradient controls, and valid-token denominators. It must declare whether actor and critic share parameters and how their losses are weighted. Diagnostic logs should include RM reward, shaped return, policy loss, value loss, advantage statistics, ratio and clip fractions, KL statistics, response length, entropy, finite-value flags, and independent evaluation results.

Finally, a resumable checkpoint requires more than policy weights. It must preserve the critic and any shared-head state, optimizer and scheduler states, RM and reference identities, prompt-data cursor, rollout-buffer semantics when resuming mid-iteration, random states, counters, and the full objective configuration. Chapter 11's distinction between a model snapshot and a truly resumable training state applies unchanged to RLHF.

= Summary <sec-rlhf-summary>

RLHF turns a preference-trained RM into an optimization signal for an autoregressive policy. A prompt plus generated prefix is the state, the next token is the action, and the completed response is a trajectory scored by the RM. Because the policy samples discrete tokens and reward is commonly delayed, policy gradients use sampled returns to change token probabilities. The critic supplies a value baseline, and GAE converts shaped rewards and value predictions into lower-variance advantage estimates.

PPO uses the old-policy probability ratio and a clipped surrogate to restrain each update on fresh Rollouts. The reference-model KL term serves a different purpose: it anchors the evolving policy to the SFT distribution and makes the reward-versus-drift trade-off explicit through $beta$. Neither mechanism makes the RM correct. Policy loss, value loss, RM score, and external quality are distinct signals that must be monitored separately.

The practical loop is therefore a versioned on-policy system: sample prompts, generate responses, score them, estimate returns and advantages, update the policy and critic cautiously, evaluate externally, and repeat. Its central risk is reward overoptimization, in which a policy improves the learned proxy while degrading the intended behavior. The next post-training methods can be understood only after this distinction between preference-derived reward, policy optimization, and independent evaluation is clear.

#heading(level: 1, numbering: none, outlined: false)[References]
#chapter-bibliography("/references/handbook.bib")
