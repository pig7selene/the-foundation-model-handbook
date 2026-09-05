#import "../../../templates/typst/chapter.typ": technical-chapter, abstract, academic-table, chapter-bibliography

#show: technical-chapter.with(
  title: [Feed-Forward Networks, Normalization, and Residual Connections],
)

#abstract[
  Attention moves information across token positions, but it is not the only source of computation in a Transformer block. This chapter develops the position-wise Feed-Forward Network as a learned expansion, nonlinear transformation, and contraction; derives gated variants and SwiGLU; and compares LayerNorm with RMSNorm at the level of their statistics and axes. It then treats residual connections, normalization placement, initialization, and gradient flow as one coupled design problem, concluding with parameter, compute, activation, and numerical implementation contracts relevant to modern decoder-only models.
]

= Introduction <sec-ffn-introduction>

A decoder-only Transformer block alternates two qualitatively different operations. Self-Attention mixes information across positions, whereas the Feed-Forward Network (FFN) transforms the features of each position independently. Normalization determines the scale at which each sublayer reads the residual stream, and the residual connection determines how the resulting update is accumulated. These components are sometimes presented as architectural details surrounding attention. In a deep language model they are instead part of the mechanism that makes repeated sequence mixing expressive and trainable.

Let $H in R^(B times T times d)$ denote a residual-stream tensor with batch size $B$, sequence length $T$, and model width $d$. The FFN intermediate width is denoted by $m$. Every operation considered here preserves the outer shape $B times T$; normalization and the FFN act independently on the final feature axis, while residual addition requires the sublayer output to return to width $d$.

= Position-Wise Feed-Forward Networks <sec-position-wise-ffn>

== Expansion, Nonlinearity, and Contraction <sec-ffn-expansion>

The original Transformer applies the same two-layer network to every sequence position @vaswani2017attention. For a single feature vector $x in R^d$, a conventional FFN is

$
  op("FFN")(x)
  = phi(x W_("up") + b_("up")) W_("down") + b_("down"),
$ <eq-vanilla-ffn>

with $W_("up") in R^(d times m)$, $b_("up") in R^m$, $W_("down") in R^(m times d)$, and $b_("down") in R^d$. Applied to $H$, the first affine map produces a tensor in $R^(B times T times m)$; the second returns it to $R^(B times T times d)$. The weights are shared across all positions and batch items. Consequently, @eq-vanilla-ffn mixes features but never token positions.

The width $m$ is usually larger than $d$. Expansion gives the nonlinearity a higher-dimensional feature space in which to form an update; contraction makes that update compatible with the residual stream. Without the nonlinear function $phi$, the two affine maps would collapse into a single affine transformation, apart from their combined bias, and the expanded representation would add no expressive depth.

Ignoring biases, @eq-vanilla-ffn contains $2 d m$ parameters. For each token it also requires approximately $2 d m$ scalar multiply-accumulates: $d m$ for each projection. If one multiplication and one addition are counted as two floating-point operations, this is approximately $4 d m$ FLOPs. Stating the counting convention matters because systems reports use both conventions.

== ReLU, GELU, and SiLU <sec-ffn-activations>

The original Transformer used the Rectified Linear Unit,

$
  op("ReLU")(z) = max(0, z),
$ <eq-relu>

applied elementwise. Later language models commonly use smoother functions. The Gaussian Error Linear Unit is

$
  op("GELU")(z) = z Phi(z),
$ <eq-gelu>

where $Phi$ is the standard normal cumulative distribution function @hendrycks2016gelu. The Sigmoid Linear Unit, usually implemented as SiLU or Swish, is

$
  op("SiLU")(z) = z sigma(z),
$ <eq-silu>

where $sigma(z) = 1 / (1 + exp(-z))$. ReLU removes every negative input exactly. GELU and SiLU instead attenuate negative inputs smoothly and retain a small nonzero response over part of the negative half-line. The choice changes optimization and representation behavior, but its effect cannot be reduced to smoothness alone; model scale, initialization, precision, and the surrounding gated architecture also matter.

= Gated Feed-Forward Networks and SwiGLU <sec-gated-ffn>

== Multiplicative Gating <sec-multiplicative-gating>

A gated FFN replaces the single transformed branch in @eq-vanilla-ffn with the elementwise product of two learned projections. In a bias-free form,

$
  op("GFFN")(x)
  = (phi(x W_g) ⊙ (x W_u)) W_d,
$ <eq-gated-ffn>

where $W_g, W_u in R^(d times m)$, $W_d in R^(m times d)$, and $⊙$ denotes the Hadamard product. One branch determines a data-dependent gate and the other supplies candidate features. Because both branches depend on $x$, the product introduces a multiplicative interaction that is absent from a single activation applied after one projection.

SwiGLU chooses SiLU for the gated branch:

$
  op("SwiGLU")(x)
  = (op("SiLU")(x W_g) ⊙ (x W_u)) W_d.
$ <eq-swiglu>

Shazeer compared several GLU variants in Transformer FFNs and introduced the SwiGLU naming and formulation @shazeer2020glu. Modern decoder-only models often combine SwiGLU with Pre-Norm and RMSNorm; LLaMA is a prominent documented example @touvron2023llama. This convention is empirical rather than a mathematical requirement: the gate, normalization, residual organization, and intermediate width remain separable architectural choices.

== Parameter-Matched Intermediate Width <sec-gated-width>

The extra projection changes the appropriate comparison of widths. A conventional FFN of width $m_v$ has approximately $2 d m_v$ weights, whereas a gated FFN of width $m_g$ has approximately $3 d m_g$. Matching their leading parameter counts gives

$
  2 d m_v approx 3 d m_g
  quad => quad
  m_g approx frac(2, 3) m_v.
$ <eq-parameter-matched-gated-width>

Thus a vanilla expansion $m_v = 4 d$ corresponds to a gated width near $8 d / 3$ at the same leading parameter budget. Implementations usually round $m_g$ to a hardware-friendly multiple. Comparing a $4 d$ vanilla FFN directly with a $4 d$ SwiGLU block silently gives the gated model roughly fifty percent more projection parameters and multiply-accumulates.

= Normalization over the Feature Axis <sec-normalization>

Normalization in a Transformer is applied independently to each token vector. It does not aggregate statistics across the batch or across sequence positions. Let $x in R^d$, let $epsilon > 0$ be a numerical stabilizer, and let $gamma, beta in R^d$ be learned elementwise affine parameters where present.

== LayerNorm <sec-layernorm>

LayerNorm computes the feature mean and population variance

$
  mu(x) = frac(1, d) sum_(i=1)^d x_i,
  quad
  sigma^2(x) = frac(1, d) sum_(i=1)^d (x_i - mu(x))^2,
$ <eq-layernorm-statistics>

then returns

$
  op("LayerNorm")(x)_i
  = gamma_i frac(x_i - mu(x), sqrt(sigma^2(x) + epsilon)) + beta_i.
$ <eq-layernorm>

The mean and variance are recomputed for each token at both training and inference time; they do not depend on batch-level running statistics @ba2016layernorm. LayerNorm is invariant to a uniform shift of all features before the learned affine transformation and is also invariant to positive rescaling, up to the effect of $epsilon$.

== RMSNorm <sec-rmsnorm>

RMSNorm removes mean subtraction and normalizes by the root mean square,

$
  op("rms")(x) = sqrt(frac(1, d) sum_(i=1)^d x_i^2 + epsilon),
$ <eq-rms-statistic>

$
  op("RMSNorm")(x)_i = gamma_i frac(x_i, op("rms")(x)).
$ <eq-rmsnorm>

In its usual form there is no learned bias $beta$. RMSNorm therefore uses the second raw moment rather than the variance: it provides rescaling invariance but not recentering invariance. Zhang and Sennrich proposed RMSNorm as a computationally simpler alternative to LayerNorm and found recentering dispensable in their studied settings @zhang2019rmsnorm. That empirical result does not make the two transformations identical; a feature vector with a large nonzero mean is treated differently by @eq-layernorm and @eq-rmsnorm.

#figure(
  academic-table(
    columns: (1.05fr, 1.55fr, 1.45fr, 1.05fr),
    align: (left, left, left, center),
    header: (
      [*Method*], [*Statistic*], [*Core transform*], [*Typical affine parameters*],
    ),
    rows: (
      [LayerNorm], [Mean and centered variance], [Center, then divide by standard deviation], [$gamma, beta$],
      [RMSNorm], [Root second moment], [Divide by RMS without centering], [$gamma$],
    ),
  ),
  caption: [LayerNorm and RMSNorm operate over the feature axis of each token vector. The table describes their standard forms; implementation variants may alter the affine parameters.],
) <tab-normalization-comparison>

== Numerical Implementation <sec-normalization-numerics>

Both normalizers reduce $d$ values into one scale, so their implementation must balance bandwidth, precision, and kernel fusion. The stabilizer $epsilon$ belongs inside the square root in @eq-layernorm and @eq-rms-statistic. In mixed-precision training, it is common to accumulate the reduction in higher precision than the stored activations and then cast the normalized result as required by the surrounding kernel. A mathematically equivalent rearrangement is not automatically numerically equivalent: computing the variance as $E[x^2] - E[x]^2$, for example, can suffer cancellation when the mean is large.

The affine parameters are broadcast over the $B$ and $T$ axes. A frequent implementation error is to normalize over more than the last feature axis, which couples tokens or batch items and changes the model. Another is to insert a second, unintended $epsilon$ or to use different values across training and inference. These choices are part of the checkpoint's executable specification.

= Residual Connections and Block Ordering <sec-residual-ordering>

== The Residual Stream <sec-residual-stream>

Let $F_l$ denote the learned transformation of sublayer $l$, including either attention or an FFN. A residual update has the shape contract

$
  H^(l) = H^(l-1) + Delta H^l,
  quad
  H^(l-1), Delta H^l in R^(B times T times d).
$ <eq-residual-shape-contract>

The tensor $H$ is called the residual stream because every sublayer reads from and writes an update to a common width-$d$ representation. The additive path can preserve existing information while the learned branch contributes a correction. It also imposes a strict interface: an FFN may expand internally to $m$, but its contraction must return to $d$ before residual addition.

== Post-Norm and Pre-Norm <sec-pre-post-norm>

The original Transformer used Post-Norm sublayers @vaswani2017attention,

$
  y = N(x + F(x)),
$ <eq-post-norm-detailed>

where normalization follows residual addition. A Pre-Norm sublayer instead computes

$
  y = x + F(N(x)).
$ <eq-pre-norm-detailed>

The two forms are not interchangeable. Pre-Norm presents a normalized input to $F$ while leaving an exact identity path from $x$ to $y$. Post-Norm normalizes the combined stream, so the identity branch also passes through the Jacobian of $N$. GPT-2 documented moving normalization to the input of each sub-block and adding a final normalization after the stack @radford2019language; Pre-Norm families have since become common in decoder-only models.

For a Pre-Norm stack, repeated substitution exposes the additive structure

$
  H^(L) = H^(0) + sum_(l=1)^L F_l(N_l(H^(l-1))).
$ <eq-pre-norm-additive-stack>

This equation does not imply that layers are independent: each update depends on the accumulated stream. It does show that a forward identity route persists through arbitrary depth, and the backward Jacobian of one sublayer has the form

$
  frac(partial y, partial x) = I + J_F J_N.
$ <eq-pre-norm-jacobian>

For Post-Norm, the corresponding local Jacobian is $J_N(I + J_F)$. The placement of $J_N$ changes how gradients compose across depth, which is one reason normalization location affects optimization rather than merely activation scale.

= Initialization, Gradient Flow, and Stability <sec-initialization-gradient-flow>

== Variance-Preserving Scale <sec-variance-initialization>

Initialization should keep activations and gradients in a useful numerical range before learning has organized the network. Consider a bias-free projection $y = x W$ with $W in R^(d_("in") times d_("out"))$. If the components of $x$ and $W$ are independent, zero mean, and have variances $q$ and $s^2$, then

$
  op("Var")(y_j) approx d_("in") s^2 q.
$ <eq-linear-variance>

Choosing $s^2$ on the order of $1 / d_("in")$ therefore preserves variance to first order. Activations and gates change the constant, and correlations accumulated during training invalidate the independence assumptions, so @eq-linear-variance is an initialization heuristic rather than a guarantee. It is nevertheless a useful audit: a projection whose standard deviation ignores fan-in can make scale grow or shrink immediately with width.

== Residual Accumulation with Depth <sec-residual-initialization>

Residual addition creates a second scale problem. Under the simplifying assumption that the residual branch updates are zero mean and uncorrelated with the incoming stream,

$
  op("Var")(H^(L))
  approx op("Var")(H^(0)) + sum_(l=1)^L op("Var")(Delta H^l).
$ <eq-residual-variance-growth>

If every branch contributes the same fixed variance, the residual-stream variance grows on the order of $L$. Practical architectures counter this tendency through normalization, fan-aware initialization, and sometimes depth-dependent scaling of residual projections. GPT-2, for example, reports scaling residual-layer weights at initialization by $1 / sqrt(N)$ for $N$ residual layers to account for accumulation with depth @radford2019language. The exact factor depends on what is counted as a residual branch and where it is applied; copying a constant without matching the architecture defeats the purpose of the derivation.

== Interpreting Gradient-Flow Claims <sec-gradient-flow-claims>

The identity term in @eq-pre-norm-jacobian supplies a direct gradient contribution, but it does not make optimization automatically stable. The learned Jacobian can still produce large updates, normalization can amplify small-variance inputs, and finite-precision arithmetic, optimizer state, learning rate, and data distribution all interact with the architecture. Conversely, Post-Norm can train successfully when initialization and learning-rate warmup are chosen appropriately.

Xiong et al. analyze gradients at initialization and connect Post-Norm's placement to large output-layer gradients and the usefulness of warmup, while finding better-behaved initial gradients for their Pre-Norm setting @xiong2020layernorm. The correct conclusion is conditional: normalization placement changes the gradient paths and therefore the optimization regime. It is not a theorem that every Pre-Norm model is stable or that every Post-Norm model is unstable.

= Parameter, Compute, and Memory Accounting <sec-ffn-resource-accounting>

The leading costs of the FFN are summarized below. Biases, normalization parameters, elementwise activations, and residual additions are lower order in parameter count relative to the dense projections. The activation-memory column describes logical intermediates; fused kernels, recomputation, and checkpointing can reduce what must be retained.

#figure(
  academic-table(
    columns: (1.1fr, 1.2fr, 1.25fr, 1.55fr),
    align: (left, center, center, left),
    header: (
      [*FFN form*], [*Leading parameters*], [*MACs per token*], [*Width-$m$ intermediates*],
    ),
    rows: (
      [Vanilla], [$2 d m$], [$2 d m$], [One projected tensor before contraction],
      [Gated / SwiGLU], [$3 d m$], [$3 d m$], [Two projections before elementwise gating],
    ),
  ),
  caption: [Leading projection costs for bias-free FFNs. One multiply-accumulate is counted as one MAC; a FLOP convention that counts multiplication and addition separately doubles these values.],
) <tab-ffn-costs>

For a batch of sequences, multiply the per-token arithmetic by $B T$. A vanilla FFN therefore performs on the order of $2 B T d m$ MACs, while a gated FFN performs $3 B T d m$. Normalization and residual addition require only $O(B T d)$ arithmetic, but they read and write full residual-stream tensors and can be limited by memory bandwidth. Operator fusion is valuable because it avoids materializing every normalized vector, bias result, activation, gate, and residual update as a separate memory round trip.

The FFN often owns more parameters than attention within a dense Transformer layer. With $m = 4 d$, a vanilla FFN has approximately $8 d^2$ weights, compared with roughly $4 d^2$ for full-width query, key, value, and output projections in conventional MHA. This comparison concerns projection weights, not total runtime: at long sequence lengths, attention's pairwise position cost can dominate even when its parameter count is smaller.

= Implementation Contracts <sec-ffn-implementation-contracts>

A reliable implementation should make shapes and axes explicit at module boundaries. Given input $B times T times d$, an FFN must return exactly $B times T times d$; normalization must reduce only the feature axis; and residual addition must not rely on accidental broadcasting. For gated FFNs, checkpoint conversion must preserve the identity and ordering of the gate and value projections. Swapping them changes @eq-swiglu because SiLU is applied to only one branch.

Numerical checks should include finite outputs for nearly constant vectors, agreement between training and inference normalization, and a reference comparison in higher precision. Initialization tests can measure empirical output variance for random inputs and verify that every intended residual projection received its depth scaling. These checks do not replace end-to-end training, but they catch silent specification errors before optimizer behavior is blamed for an incorrect tensor program.

= Summary <sec-ffn-summary>

The position-wise FFN expands each token representation, applies nonlinear or gated feature interactions, and contracts the result back to the residual width. SwiGLU adds a learned multiplicative gate and therefore requires a narrower intermediate dimension when compared at fixed parameter count. LayerNorm centers and rescales each token vector, whereas RMSNorm rescales by its second raw moment without centering. Residual connections accumulate sublayer updates into a shared stream, and the placement of normalization determines whether the identity path itself passes through a normalizer.

These choices jointly determine more than a block diagram. They set the leading FFN parameter and compute budget, the shape of saved activations, the numerical behavior of reductions, and the routes by which gradients traverse depth. Initialization and residual scaling should therefore be derived from the actual block organization rather than selected as isolated defaults.

#heading(level: 1, numbering: none, outlined: false)[References]
#chapter-bibliography("/references/handbook.bib")
