#import "../../../templates/typst/chapter.typ": technical-chapter, abstract, academic-table, chapter-bibliography

#show: technical-chapter.with(
  title: [Preference Data and Reward Modeling],
)

#abstract[
  Supervised Fine-Tuning teaches a language model to imitate demonstrations, but many desired properties of an answer are comparative: one response can be more helpful, faithful, safe, or appropriately concise than another. This chapter introduces preference data as a collection of such judgments and develops the Reward Model used by classical RLHF to convert pairwise comparisons into a scalar ranking signal. It derives the Bradley-Terry likelihood, distinguishes a reward score from a calibrated probability or universal utility, examines collection and evaluation protocols, and explains the biases, distribution shifts, and reward-hacking risks that constrain later policy optimization.
]

= Introduction <sec-preference-introduction>

Chapter 12 described Supervised Fine-Tuning (SFT) as maximum-likelihood training on demonstrations of desired responses. A demonstration answers the question, "what response should follow this prompt?" It does not necessarily express the finer judgment that arises when several acceptable responses differ in factuality, relevance, safety, brevity, tone, or reasoning quality. For many open-ended prompts, there is no single target whose token sequence exhausts that judgment.

Preference learning supplies a different supervision primitive. Given a prompt and two candidate responses, an annotator identifies the one that better satisfies a stated criterion. The comparison is cheaper to express than a complete ideal response in many settings and preserves a useful fact: the label is relative to the particular prompt, candidates, rubric, and annotator population. Early work on learning from human preferences used comparisons to infer goals that were difficult to encode as a hand-written reward function @christiano2017preferences. In language modeling, the familiar classical pipeline is

$
  "Pretraining" arrow "SFT" arrow "Preference Data"
  arrow "Reward Modeling" arrow "RLHF".
$ <eq-preference-pipeline>

Pretraining provides a broad language prior; SFT supplies an instruction-following policy and often the source of initial candidates; preference data records comparative judgments; a Reward Model (RM) learns to rank candidate responses; and a later policy-optimization stage uses that learned signal. This chapter ends at the RM. It does not derive PPO, nor does it develop DPO or GRPO beyond noting that they address later stages of the post-training sequence.

= From Demonstrations to Preferences <sec-preference-demonstrations>

An SFT record normally contains a context $x$ and a demonstrated continuation $y^*$, then optimizes the likelihood $p_theta(y^* | x)$. The supervision says that the tokens of $y^*$ are desirable targets under the training serialization and loss mask. A preference record instead contains a prompt $x$, a chosen response $y_"w"$, and a rejected response $y_"l"$. Its label says only that the annotator preferred $y_"w"$ to $y_"l"$ under the annotation policy. It does not claim that $y_"w"$ is globally perfect or that $y_"l"$ is unusable in every context.

This difference matters for data design. Demonstrations place probability mass directly on a response distribution. Comparisons expose a direction of improvement among responses sampled from a particular proposal distribution. A response pair can be informative even when neither member is an ideal answer: one may be factually correct but needlessly verbose, while the other is concise but omits a required constraint. The annotation rubric determines which trade-off the label represents.

The candidates should therefore be treated as part of the data-generating process, not as incidental strings. They may be sampled from an SFT checkpoint, a mixture of checkpoints, human-written alternatives, or carefully filtered synthetic generators. If the pool contains only obviously poor and obviously good completions, a reward model can achieve high held-out pair accuracy without learning the subtle distinctions encountered during later optimization. If it contains candidates from only one response style or length range, the learned ranking will encode that narrow comparison distribution.

= Preference Data and Its Collection <sec-preference-data-collection>

A practical preference record needs more provenance than the final binary label. At minimum it should identify the prompt and conversation context, candidate-generation policy and decoding settings, annotation rubric and task version, annotator or judging source, label outcome, collection time, and split assignment. For multi-turn dialogue, the complete preceding history and its Chat Template are part of $x$. As Chapter 12 emphasized, changing the serialized context changes the conditional task presented to a language model and to a judge.

#figure(
  block(width: 100%)[
    #set text(size: 9.05pt)
    #set par(justify: false, leading: 0.55em, spacing: 0pt)
    #academic-table(
      columns: (1.1fr, 1.65fr, 1.7fr),
      align: (left, left, left),
      inset: (x: 4pt, y: 2.7pt),
      header: (
        [*Collection source*], [*What the comparison can provide*], [*Principal limitation to record*],
      ),
      rows: (
        [Human annotation], [Judgment under an explicit human rubric, including domain expertise when the task requires it.], [Cost, disagreement, rubric interpretation, and limited coverage.],
        [Expert or audited annotation], [Higher-fidelity comparisons for specialized or high-stakes criteria.], [Expense and possible narrowing to the experts' operational policy.],
        [Synthetic or AI feedback], [Rapidly generated comparisons and coverage of rare or adversarial cases.], [The judge's blind spots, prompt sensitivity, and shared model-family biases.],
        [Hybrid collection], [Human anchors or audits combined with scalable generated labels.], [The route and quality controls for every label must remain recoverable.],
      ),
    )
  ],
  caption: [Preference sources differ in what they measure, not only in cost. A response pair must retain its provenance so that a later RM evaluation can distinguish human and synthetic supervision.],
) <tab-preference-data-sources>

Human annotation is not a single protocol. The task may ask an annotator to choose the more helpful response, identify the less harmful response, compare factual accuracy using supplied evidence, or break ties according to a detailed style policy. Randomizing response order avoids turning a fixed presentation position into an accidental label. Blinded candidate identities avoid source-model reputation effects. Multiple labels for selected examples make disagreement visible rather than silently collapsing it into a single asserted preference. InstructGPT collected rankings of model outputs after SFT as the source for its Reward Model stage @ouyang2022training.

Synthetic preference data replaces or augments human choice with a model-based judge, a rule-based verifier, or a constitution-like critique-and-revision process. It can make coverage and iteration substantially cheaper, but it is not automatically human preference data. Constitutional AI and RLAIF are examples of pipelines that use AI-generated feedback to scale parts of the supervision process @bai2022constitutional @lee2023rlaif. A synthetic label inherits the criterion, failure modes, and distributional limits of its generating judge. It should therefore record the judge revision, prompt, decoding policy, and any filtering or audit rule, just as human data records its rubric.

== Noise, Ties, and Inconsistent Preferences <sec-preference-noise>

Preference labels are observations, not ground truth utilities. Annotators can misread a response, apply the rubric differently, or reasonably disagree because the comparison presents a real trade-off. Inconsistency also arises when a label depends on information absent from the stored prompt, when candidates differ in subtle ways that the rubric does not settle, or when the same comparison is presented with a different order or framing.

A binary dataset often resolves a tie by asking for a forced choice. That can be operationally convenient, but it turns ambiguity into label noise. Systems that collect ties, strength-of-preference labels, or repeated annotations can represent more of the uncertainty, though their modeling and aggregation choices must then be explicit. The basic pairwise objective below deliberately models a binary observed choice; it does not prove that every human has one stable, transitive ranking over all possible responses.

= Pairwise Preference Modeling <sec-pairwise-preference-modeling>

Let $r_psi(x, y)$ be a scalar score assigned by an RM with parameters $psi$ to response $y$ in context $x$. The Bradley-Terry model represents the probability that the chosen response wins a pairwise comparison as

$
  P(y_"w" succ y_"l" | x)
  = sigma(r_psi(x, y_"w") - r_psi(x, y_"l")),
  quad
  sigma(z) = frac(1, 1 + exp(-z)).
$ <eq-bradley-terry-preference>

The score difference, not either score in isolation, determines the probability in @eq-bradley-terry-preference. When both scores are equal, the model assigns probability $1 / 2$ to the observed ordering. Increasing the chosen-minus-rejected margin raises the probability. Reversing the candidates negates the margin and swaps the probability. This is a useful statistical model of noisy comparative choices, not evidence that human judgments literally arise from a logistic latent utility.

For a dataset $cal(P) = {(x_i, y_"w"^(i), y_"l"^(i))}_(i=1)^N$, maximum likelihood yields the negative log-likelihood

$
  cal(L)_"RM"(psi)
  = -frac(1, N) sum_(i=1)^N
    log sigma(
      r_psi(x_i, y_"w"^(i)) - r_psi(x_i, y_"l"^(i))
    ).
$ <eq-reward-model-nll>

Writing $Delta_i = r_psi(x_i, y_"w"^(i)) - r_psi(x_i, y_"l"^(i))$ makes one example's contribution

$
  -log sigma(Delta_i) = log(1 + exp(-Delta_i)).
$ <eq-preference-logistic-loss>

Thus a correctly ordered pair with a large positive margin contributes little loss; an incorrectly ordered pair with a negative margin contributes heavily. The derivative with respect to $Delta_i$ is $sigma(Delta_i) - 1$, so uncertain or wrong pairs exert stronger pressure than already confidently correct pairs. In practice, this loss is evaluated with a numerically stable log-sigmoid or log-sum-exp form, following the stability principles of Chapter 8.

The formulation also reveals an identifiability limit. For any function $a(x)$,

$
  (r_psi(x, y_"w") + a(x)) - (r_psi(x, y_"l") + a(x))
  = r_psi(x, y_"w") - r_psi(x, y_"l").
$ <eq-reward-score-shift-invariance>

The pairwise likelihood cannot determine a prompt-dependent common offset. It mainly learns relative ordering inside the comparison task. This is why an RM score should not be read as a portable absolute measurement of helpfulness, safety, or human welfare.

== Scores, Probabilities, and Rankings <sec-reward-score-interpretation>

Three distinctions prevent common misreadings. First, *classification accuracy* asks how often an RM ranks the chosen candidate above the rejected candidate in a held-out labeled pair. It is a useful diagnostic, but it is not reward quality in full: it may ignore ties, calibration, distribution shift, adversarial candidates, and the behavior induced when a policy searches for high scores.

Second, a *reward score* $r_psi(x, y)$ is a model output. The preference *probability* is obtained only after comparing two scores through @eq-bradley-terry-preference. A score of $3$ is not a probability of $3$ percent, nor does a score twice as large mean that a response is twice as good. Its scale can change with architecture, training loss, normalization, regularization, and the candidate distribution.

Third, pairwise data naturally identifies a ranking relation rather than a calibrated global score. *Absolute scoring* would require labels with a declared reference scale, such as a rubric-defined rating, and even then demands separate calibration analysis. The Bradley-Terry objective makes a local comparative claim: for this context and pair, one response is predicted to be more likely to win. It does not create a universal ruler across unrelated prompts.

= Reward Models as Sequence-Level Scorers <sec-reward-models>

For a decoder-only RM, the context and candidate response are serialized into one sequence, often using the same tokenizer and Chat Template family as the policy. The Transformer processes the entire prefix causally, and a scalar head maps a chosen hidden state to a sequence-level score. If $h_T(x,y)$ denotes the hidden state at the declared terminal response position, a common form is

$
  r_psi(x, y) = w^T h_T(x, y) + b,
$ <eq-sequence-level-reward>

where $psi$ includes the backbone parameters and scalar-head parameters $(w, b)$. Other readout conventions are possible, but the terminal position, end-of-response token handling, padding policy, and truncation rule change the actual function being learned. A reward is called *sequence-level* here because it evaluates a completed candidate response as one object; it is not a token-level next-token likelihood and it need not decompose into rewards for individual tokens.

The RM commonly starts from a pretrained or SFT-capable backbone because it must understand the prompt, response, and their relationship. This initialization is not a guarantee of correct judgments. The scalar head and fine-tuning objective can exploit superficial correlates in the comparison data. Stiennon et al. trained a scalar reward predictor from human summary comparisons and then used it as the reward signal for a later policy stage, making the separation between reward-model fitting and policy optimization explicit @stiennon2020summarize.

== Calibration and Score Normalization <sec-reward-calibration>

Calibration concerns whether predicted pairwise probabilities correspond to observed frequencies on an appropriate held-out distribution. If the RM assigns roughly $0.8$ to many comparisons of a defined type, then roughly $80$ percent of those chosen responses should win under the same labeling policy. A temperature $tau > 0$ can rescale the margin for a held-out calibration procedure,

$
  P_tau(y_"w" succ y_"l" | x)
  = sigma(frac(r_psi(x, y_"w") - r_psi(x, y_"l"), tau)).
$ <eq-reward-temperature-calibration>

Calibration does not repair a biased preference dataset, an out-of-distribution response, or a misspecified criterion. Nor does it give the raw reward a universal meaning. Mean-variance normalization can make reward magnitudes more convenient for a later optimizer, but it is a training-interface convention, not a discovery of the true scale of human values. The denominator and reference distribution used for any normalization must be stored with the model.

= Bias, Distribution Shift, and Reward Hacking <sec-reward-failures>

Reward Models learn patterns predictive of labels, which need not coincide with the intended property. A dataset can reward longer answers because annotators interpret detail as effort, even when extra length obscures the answer. Stiennon et al. explicitly controlled summary length because length can confound preference judgments @stiennon2020summarize. A model can similarly learn a preference for polished formatting, a particular refusal style, hedging, citations, or a judge-favored dialect. Such style correlates are not always wrong; they become harmful when they displace task correctness or user intent.

Annotator disagreement is another source of bias rather than merely an inconvenience. A single aggregate label can erase legitimate variation in how people weigh harmlessness, directness, technical detail, or tone. Dataset policy may require aggregation, but evaluation should retain enough metadata to ask where agreement is low, which rubric version applies, and whether the RM amplifies a majority preference beyond its evidential support.

*Distribution shift* occurs when the prompts, candidates, languages, response lengths, or behaviors evaluated by the RM differ from those on which it was trained. Later policy optimization creates a particularly important shift: it deliberately searches for responses with high predicted reward and can therefore produce unusual candidates. A held-out comparison sampled from the original candidate generator may not expose errors in that searched region.

*Reward hacking* or *reward overoptimization* is the resulting gap between a rising proxy score and the intended external judgment. The policy is not required to be deceptive for this gap to occur; it need only find a feature that the RM rewards more than humans do. Reward-model ensembles can reduce some idiosyncratic errors, but they do not remove shared blind spots. Eisenstein et al. show that RMs with similar in-distribution behavior can diverge under alignment because of underspecification and distribution shift @eisenstein2023helping. This is a reason to preserve human and task-grounded evaluation after RM training, not a reason to treat an ensemble score as an oracle.

= Evaluating Reward Models <sec-reward-evaluation>

The first evaluation is pairwise held-out accuracy: does $r_psi(x, y_"w") > r_psi(x, y_"l")$ for a protected set of comparisons? Report the split construction, label aggregation, tie policy, candidate sources, and confidence intervals or uncertainty estimates. Accuracy should be broken down by task, domain, response length, safety criterion, and source model when those attributes are available. A single aggregate number can conceal a model that ranks ordinary chat well while failing factual, adversarial, or long-context comparisons.

Pairwise accuracy alone cannot test score calibration, robustness, or the downstream consequences of optimization. A complementary evaluation includes calibration curves for pairwise probabilities, response-order swaps, length-controlled comparisons, repeated labels, and targeted contrast sets where one response has a verifiable flaw. RewardBench was introduced as a benchmark suite containing prompt-chosen-rejected trios across chat, reasoning, and safety, including structured and out-of-distribution cases @lambert2024rewardbench. Its role in a workflow is illustrative: evaluate the ranking function on disjoint, diagnostic tasks rather than trusting an in-distribution training loss.

The final evaluation question is behavioral. If an RM is later used for reranking or policy optimization, compare high-RM-score outputs with independent human or task-grounded judgments. Track the gap between proxy reward, held-out RM accuracy, and the external criterion as optimization strength changes. This does not enter PPO mechanics; it establishes the validation boundary that the next RLHF chapter must respect.

= Implementation Contracts <sec-reward-implementation-contracts>

The data contract should version the prompt and conversation serialization, tokenizer, special tokens, Chat Template, candidate generator, decoding settings, annotation rubric, annotator or judge source, candidate order randomization, label aggregation rule, and split assignment. It should preserve both candidates even when a pair is discarded, together with the reason for discard. Duplicate prompts, near-duplicate responses, malformed boundaries, unsupported roles, missing terminal tokens, and labels without a declared policy must be rejected or handled explicitly.

The tensor contract should identify the token span used by the scalar readout in @eq-sequence-level-reward, attention and padding masks, pair packing convention, response-order convention, and the loss reduction in @eq-reward-model-nll. Tests should swap $y_"w"$ and $y_"l"$ and verify that the signed margin reverses. They should verify that padding and truncation cannot change a response's readout position unintentionally, and that each pair contributes exactly once to the declared denominator.

The evaluation contract should separate training, validation, calibration, and external-test comparisons by prompt and by sufficiently strong near-duplicate controls. It should record pairwise accuracy, margin distributions, calibration diagnostics, length and source-model slices, disagreement statistics, and any human audit. If the RM is consumed by a later optimizer, checkpoints must retain the tokenizer, template, scalar-head convention, normalization metadata, and provenance needed to reproduce scores. A scalar head without this interface and data-policy information is not a fully specified reward signal.

= Summary <sec-preference-summary>

Preference data complements SFT by expressing a relative judgment between candidate responses rather than a single demonstrated continuation. In the classical pipeline, SFT supplies an initial instruction-following policy, comparison data records which candidates better satisfy a rubric, and a Reward Model turns those comparisons into a reusable ranking signal before RLHF optimizes a policy against it.

The Bradley-Terry model converts a reward-score difference into a pairwise preference probability and yields a logistic negative log-likelihood for RM training. Its score is not an absolute utility or a probability by itself: pairwise labels mainly identify local rankings, and common score shifts remain unidentifiable. Calibration and normalization can make the output better behaved for a declared distribution and interface, but they cannot repair a misaligned criterion.

Reward quality is constrained by the data and search distribution. Human and synthetic preferences each carry provenance and bias; length, style, disagreement, and distribution shift can make high pairwise accuracy misleading. The most important discipline before policy optimization is to keep external, human or task-grounded evaluation separate from the learned proxy. The next chapter can then introduce RLHF as an optimization problem with an explicitly limited reward signal rather than as a mechanism that makes the RM authoritative.

#heading(level: 1, numbering: none, outlined: false)[References]
#chapter-bibliography("/references/handbook.bib")
