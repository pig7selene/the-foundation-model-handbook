#import "../../../templates/typst/chapter.typ": technical-chapter, abstract, academic-table, chapter-bibliography

#show: technical-chapter.with(
  title: [Transformer Architecture],
)

#abstract[
  A decoder-only Transformer maps a batch of token IDs to vocabulary logits through a sequence of shape-preserving residual updates. This chapter gives the complete high-level computation graph before treating attention, positional encoding, normalization, and optimization in dedicated technical chapters. It defines the hidden-state and tensor-shape contracts of the model, contrasts Pre-Norm and Post-Norm block organization, and identifies the parameter, activation, and sequence-length terms that dominate later systems analysis.
]

= Introduction <sec-introduction>

A decoder-only Transformer is often introduced as a stack of Self-Attention layers. That description is useful but incomplete. A language model has a stable external contract: it receives a tensor of token IDs and produces one vocabulary-sized vector of logits at every position. Attention is only one transformation in the path between those two objects. The remaining transformations determine where normalization acts, how information is accumulated, which tensor dominates activation memory, and how the output vocabulary is scored.

This chapter gives that path a precise, high-level form. The aim is not to rederive attention, positional encoding, normalization, or feed-forward networks in isolation. Instead, the chapter establishes the computation graph that makes those details meaningful: a sequence enters a residual stream, each block writes updates to that stream, and a final projection turns the resulting states into logits for an autoregressive language-model objective.

= Decoder-Only Transformer Architecture <sec-decoder-only-architecture>

== From Token IDs to Hidden States <sec-transformer-inputs>

Consider a batch of $B$ token sequences, each represented to length $T$. Let

$
  X in {0, dots, |cal(V)| - 1}^(B times T)
$ <eq-transformer-token-ids>

be the integer token-ID tensor, let $d$ denote the model dimension, and let $E in R^(|cal(V)| times d)$ be the token-embedding matrix. Embedding lookup produces a rank-three tensor

$
  H^(0) = E[X] in R^(B times T times d).
$ <eq-transformer-embedding-tensor>

The superscript identifies depth rather than a power. Thus $H^(0)$ is the sequence representation before the first Transformer block, while $H^(ell)$ will denote the representation after block $ell$. If the architecture uses additive absolute positions, the input is instead $H^(0) = E[X] + P$, where $P in R^(T times d)$ is broadcast across the batch. Other positional schemes modify later computations rather than this initial tensor. In every case, the central object is the same: for batch item $b$ and position $t$, $H^(0)_(b,t,:)$ is a length-$d$ hidden-state vector.

The distinction between IDs and states is fundamental. $X$ is discrete and has no useful local geometry: token ID 800 is not intrinsically closer to ID 801 than to ID 7. Each slice of $H^(0)$ is continuous and trainable. Subsequent layers can mix information across positions and features without changing the sequence length.

== Stacked Blocks and Causal Processing <sec-decoder-only-stack>

Let $L$ be the number of Transformer blocks. A decoder-only model applies a sequence of functions $F_1, dots, F_L$ to the input representation:

$
  H^(ell) = F_ell(H^(ell - 1)), quad ell in {1, dots, L}.
$ <eq-transformer-stack>

Each $F_ell$ preserves the shape $B times T times d$. The expression *decoder-only* refers to the causal dependency pattern within these functions: the state at position $t$ may draw on positions at most $t$, but not on later tokens. It does not mean that the block processes positions one at a time during training. The full tensor is usually evaluated in parallel under a causal Attention Mask. The sequential aspect of generation emerges only when the model is used to append a new token to an existing prefix.

The original Transformer paired an encoder stack with a decoder stack for sequence-to-sequence tasks @vaswani2017attention. A Foundation Model used for Causal Language Modeling retains the masked, autoregressive branch and omits encoder-decoder cross-attention. This reduction creates a uniform contract: a single stream of token positions is transformed repeatedly, then scored against the vocabulary. GPT-2 is an influential example of this decoder-only pattern @radford2019language.

#let architecture-box(body, width: 74mm) = block(
  width: width,
  inset: (x: 8pt, y: 5pt),
  stroke: 0.6pt + black,
  radius: 1pt,
)[#align(center)[#body]]

#figure(
  align(center)[
    #grid(
      columns: (1fr,),
      row-gutter: 3pt,
      align: center,
      architecture-box[Token IDs $X$],
      [↓],
      architecture-box[Token and positional representations $H^(0)$],
      [↓],
      architecture-box(width: 94mm)[
        *Pre-Norm Transformer block, repeated $L$ times* \
        residual stream \
        ↓ \
        Norm $->$ causal attention $->$ residual addition \
        ↓ \
        Norm $->$ feed-forward network $->$ residual addition
      ],
      [↓],
      architecture-box[Final normalization $H^(L) -> Z$],
      [↓],
      architecture-box[LM head $->$ logits $O$],
    )
  ],
  caption: [A high-level decoder-only Transformer computation path. Each block reads the current residual stream and writes an update to it.],
) <fig-decoder-only-transformer>

The diagram in @fig-decoder-only-transformer omits internal parameter matrices on purpose. It exposes the architectural invariant that matters at this stage: every block receives a hidden-state tensor and returns a tensor of the same shape. Depth is therefore a controlled composition of representation updates rather than a sequence of incompatible interfaces.

= Transformer Block <sec-transformer-block>

A standard decoder-only block has two learned sublayers: causal Self-Attention and a position-wise feed-forward network (FFN). Attention allows the representation at one position to use an allowed prefix of the sequence; the FFN applies the same nonlinear transformation independently at every position. Residual connections preserve a direct route around both sublayers, while normalization controls the scale and distribution of the vectors passed into them.

For a Pre-Norm block, write $A_ell$ for the attention transformation, $M_ell$ for the FFN transformation, and $N_ell^(a)$ and $N_ell^(m)$ for their respective normalizations. The two updates are

$
  U^(ell) = H^(ell - 1) + A_ell(N_ell^(a)(H^(ell - 1))),
$ <eq-pre-norm-attention-update>

followed by

$
  H^(ell) = U^(ell) + M_ell(N_ell^(m)(U^(ell))).
$ <eq-pre-norm-ffn-update>

These equations describe control flow, not an attempt to hide the details in notation. $A_ell$ contains the query, key, value, mask, softmax, and output operations. $M_ell$ contains the expansion, activation or gate, and contraction. Their outputs have the same final feature dimension $d$ as their inputs, so they can be added to the residual stream.

The order of the two sublayers is conventional rather than mathematically necessary. The important point is that attention and the FFN have complementary roles. Attention mixes *positions*: a token representation can incorporate information from its prefix. The FFN mixes *features* within one position: it turns the resulting feature vector into a richer nonlinear update. A model needs both forms of computation to make context-sensitive predictions without treating every feature interaction as an attention operation.

= Residual Stream <sec-residual-stream>

The phrase *residual stream* is a useful abstraction for $H^(ell)$. At any depth, it is the common sequence representation carried through the network. Attention does not replace that representation; it reads a normalized view of it and adds a context-dependent update. The FFN then reads the updated stream and adds a position-wise update. The paired updates in @eq-pre-norm-attention-update and @eq-pre-norm-ffn-update therefore describe an additive computation in which information can persist across many layers while new features are written into the same width-$d$ state.

This view has two benefits. First, it makes tensor shapes easy to audit: the residual stream remains $B times T times d$ from input to final normalization. Second, it discourages a misleading picture in which a Transformer repeatedly discards and reconstructs a sequence. Each block instead makes bounded modifications to a shared representation. The identity terms in the residual connections provide direct forward and backward routes, while the learned sublayers specialize in producing useful corrections.

= Pre-Norm and Post-Norm <sec-pre-norm-post-norm>

The original Transformer uses a Post-Norm organization, in which normalization follows each residual addition. For a generic sublayer $S$, that pattern is

$
  y = N(x + S(x)).
$ <eq-post-norm>

By contrast, Pre-Norm applies normalization to the sublayer input and leaves the residual path itself unnormalized:

$
  y = x + S(N(x)).
$ <eq-pre-norm>

Neither expression is merely a rearrangement of the other. In Post-Norm, the next sublayer receives a normalized mixture of the previous stream and the current update. In Pre-Norm, the identity path from $x$ to $y$ remains explicit, while the learned update is driven by a normalized argument. Consequently, normalization placement changes both forward activation statistics and the paths followed by gradients.

This distinction became practically consequential as Transformer stacks grew deeper. Xiong et al. analyze the initialization behavior of Pre-Norm and Post-Norm Transformers and relate Post-Norm's output-layer gradient scale to the usefulness of warmup @xiong2020layernorm. Modern decoder-only architectures commonly adopt a Pre-Norm family of designs; GPT-2, for example, places LayerNorm at the input of each sub-block @radford2019language. These observations do not make Pre-Norm universally superior. They identify a stability trade-off whose interaction with optimizer choice, residual scaling, precision, and depth must be evaluated as a system.

= From Hidden States to Logits <sec-transformer-logits>

After the final block, many decoder-only models apply a final normalization $N_f$:

$
  Z = N_f(H^(L)), quad Z in R^(B times T times d).
$ <eq-final-normalization>

The language-model head is an affine map from each final hidden vector to one score per vocabulary item. With output matrix $W_("out") in R^(d times |cal(V)|)$ and bias $b_("out") in R^(|cal(V)|)$,

$
  O = Z W_("out") + b_("out"), quad O in R^(B times T times |cal(V)|).
$ <eq-lm-head>

The element $O_(b,t,v)$ is a logit, not yet a probability: it is the unnormalized score assigned to vocabulary item $v$ at position $t$. An autoregressive objective applies a masked cross-entropy loss to these scores. At inference time, a decoding rule converts the final position's logits into a distribution or selected next token.

When input and output embeddings are tied, the model reuses the token-embedding table and takes $W_("out") = E^T$, up to the orientation convention of the implementation. Weight tying reduces parameter count and connects the geometry used to read a token with the geometry used to score it. It is an architectural choice, not an identity required by decoder-only modeling. The original Transformer discusses shared embedding and pre-softmax weights in some configurations @vaswani2017attention.

= Tensor Shapes Through the Model <sec-transformer-shapes>

The following table records the shape contracts most useful when implementing or debugging the high-level forward pass. Let $h$ be the number of query heads, $d_h$ the per-head dimension, and $d_("ff")$ the FFN intermediate width; ordinarily $d = h d_h$.

#figure(
  academic-table(
    columns: (1.65fr, 1.35fr, 2.1fr),
    align: (left, center, left),
    header: (
      [*Quantity*], [*Shape*], [*Role*],
    ),
    rows: (
    [Token IDs $X$], [$B times T$], [Discrete input indices.],
    [Embeddings / residual stream $H^(ell)$], [$B times T times d$], [Sequence representation at depth $ell$.],
    [Queries, keys, values], [$B times T times h times d_h$], [Head-partitioned attention inputs.],
    [Attention output], [$B times T times d$], [Contextual update projected back to model width.],
    [FFN intermediate], [$B times T times d_("ff")$], [Expanded position-wise representation.],
    [Logits $O$], [$B times T times |cal(V)|$], [Vocabulary scores for the language-model objective.],
    ),
  ),
  caption: [Symbolic tensor shapes in a decoder-only Transformer. The exact internal layout of heads may vary by implementation, but the residual-stream and logit contracts are stable.],
) <tab-transformer-shapes>

The table distinguishes model-width tensors from vocabulary-width tensors. Most block activations have width $d$, whereas logits have width $|cal(V)|$. For a large vocabulary, materializing logits for every position can itself be substantial. Conversely, the residual stream persists at every block and is a major source of activation memory during training. Shape reasoning is therefore not bookkeeping after the fact; it is the first step toward a reliable memory and compute model.

= Computational Structure <sec-transformer-computational-view>

At a high level, each block contains four dense projection families associated with attention: query, key, value, and output. In a conventional full-width implementation, they contribute on the order of $4d^2$ parameters per layer. The FFN contributes roughly $2d d_("ff")$ parameters for a two-projection design, or a different constant factor for gated variants such as SwiGLU. These counts are only first-order approximations, but they already explain why changing $d$, $L$, or $d_("ff")$ changes the parameter budget materially.

The principal sequence-length dependence arises in attention. For a length-$T$ sequence, each position can compare against a prefix whose total pairwise work grows on the order of $T^2 d$ for dense Self-Attention. The FFN instead performs work on each position independently, scaling on the order of $T d d_("ff")$. The relative bottleneck therefore changes with sequence length, width, batch size, hardware, and kernel implementation. Cache-aware decoding and other inference methods alter the execution regime without changing the basic architectural contract.

Activation memory has a different emphasis from parameter count. During training, backward computation requires access to selected forward intermediates or to recomputed equivalents. The residual stream alone stores $B T d$ scalars per layer before accounting for attention and FFN internals. During autoregressive decoding, the model instead retains key and value representations for prior positions in a KV Cache. The forward equations are the same in spirit, but the dominant persistent state changes with the execution regime.

= Summary <sec-transformer-summary>

A decoder-only Transformer maps token IDs to embeddings, repeatedly applies residual attention and FFN updates, normalizes the final stream, and projects each hidden state to vocabulary logits. The stack in @eq-transformer-stack supplies the architectural spine; the Pre-Norm updates in @eq-pre-norm-attention-update and @eq-pre-norm-ffn-update make the residual stream explicit; and the LM head in @eq-lm-head closes the path back to the discrete vocabulary.

The high-level view deliberately treats causal attention, positional encoding, normalization, and optimization as modules. A subsequent independent chapter can derive attention masks, head variants, and positional encodings; another can study FFNs, normalization, residual scaling, and gradient flow. The computation graph developed here supplies the common tensor contracts that those analyses require.

#heading(level: 1, numbering: none, outlined: false)[References]
#chapter-bibliography("/references/handbook.bib")
