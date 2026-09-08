#import "../../../templates/typst/chapter.typ": technical-chapter, abstract, academic-table, chapter-bibliography

#show: technical-chapter.with(
  title: [Efficient Attention and #linebreak() Head-Representation Design],
)

#abstract[
  Dense attention has two distinct efficiency problems: executing its quadratic interactions without excessive memory traffic, and retaining the Key/Value state that autoregressive decoding repeatedly reads. This chapter separates these concerns. It develops FlashAttention as an exact, IO-aware execution algorithm based on tiling and online Softmax, then compares Multi-Head Attention, Multi-Query Attention, Grouped-Query Attention, and Multi-Head Latent Attention as architectural choices for representing Key/Value information. The resulting distinction clarifies why kernel optimization and KV-state compression are complementary rather than competing interventions.
]

= Introduction <sec-efficient-attention-introduction>

Chapter 3 derived causal Self-Attention as a tensor program: queries compare with eligible keys, the resulting logits are normalized, and the probabilities aggregate values. That derivation identifies the mathematical function but does not determine how a device should execute it or how a deployed model should represent the state left behind by earlier tokens. Those are separate questions with a common practical consequence: attention can be limited by moving data even when its nominal floating-point operation count appears manageable.

This chapter therefore studies two layers of design. *FlashAttention* changes the execution algorithm for the same dense attention function. It reduces traffic between slow and fast GPU memory by processing the computation in tiles and never materializing the full attention matrix in high-bandwidth memory (HBM). *MHA*, *MQA*, *GQA*, and *MLA*, by contrast, are attention-architecture choices. They determine how many Key/Value (KV) representations exist, or how those representations are compressed, and hence change persistent KV Cache state. A model using GQA can still use FlashAttention; neither choice subsumes the other.

Let $B$ be batch size, $T$ sequence length, $L$ the number of Transformer blocks, $H_q$ the number of Query heads, $H_"kv"$ the number of KV heads, and $d_h$ the head dimension. The symbols follow Chapter 3 except that the more explicit $H_q$ and $H_"kv"$ make head sharing easier to compare. We write $b$ for bytes per stored scalar. Chapter 23 distinguishes Prefill from Decode, Chapter 24 derives KV Cache accounting, and Chapter 29 explains why the active performance bottleneck must be measured rather than inferred from one complexity term alone.

= Attention Cost Is More Than FLOPs <sec-attention-cost-revisited>

For one head, Scaled Dot-Product Attention first forms scores and then applies those weights to values:

$
  S = frac(Q K^T, sqrt(d_h)), quad
  P = op("softmax")(S), quad
  O = P V.
$ <eq-dense-attention>

For a length-$T$ sequence, the two matrix products in @eq-dense-attention require arithmetic proportional to $T^2 d_h$. Across heads, the conventional dense-attention contribution is proportional to $B T^2 H_q d_h$. The all-pairs dependency is mathematical: every legal query-key pair can affect an output. An exact kernel cannot make that dependency linear in $T$.

Hardware time nevertheless depends on more than arithmetic work. A kernel reads and writes tensors, moves them through a memory hierarchy, launches work, and sometimes synchronizes intermediate results. An implementation can have the same result and nearly the same nominal FLOPs as another while running much faster because it moves fewer bytes over the limiting boundary. This is the IO-aware perspective used by FlashAttention @dao2022flashattention.

== The Materialization Problem <sec-attention-materialization>

The following conceptual execution is useful even though framework kernels may fuse some of its steps:

$
  Q, K
  arrow S = Q K^T
  arrow P = op("softmax")(S)
  arrow O = P V.
$ <eq-naive-attention-pipeline>

If $S$ and $P$ are stored as separate dense tensors, each has roughly $B H_q T^2$ elements. The device may write scores to HBM, read them to normalize rows, write probabilities, and read probabilities again for the value product. These intermediate transfers can dominate the useful matrix arithmetic at relevant shapes. The issue is not that $S$ is conceptually invalid; it is that a full $T times T$ representation is a poor execution boundary when it must repeatedly cross a slow memory boundary.

The GPU hierarchy gives the distinction its force. Registers and on-chip SRAM, often exposed partly as shared memory, are small but fast. HBM or global device memory is much larger but slower to access. A high-level model need not depend on particular bandwidth numbers to use this principle: reuse data in fast storage while it is needed, and avoid writing a large intermediate merely because a step-by-step algebraic presentation introduced it.

= FlashAttention: Exact IO-Aware Execution <sec-flashattention>

FlashAttention partitions $Q$, $K$, and $V$ into blocks that fit in fast on-chip storage. For one query block, it streams successive Key/Value blocks, computes local scores, updates rowwise normalization statistics and output accumulators, and writes only the completed output block. The computation can be summarized as

$
  "load a Q tile"
  arrow "stream K/V tiles"
  arrow "update row statistics and output"
  arrow "write O tile".
$ <eq-flashattention-tile-flow>

The full probability matrix is not materialized in HBM. The result remains dense attention: every unmasked query-key interaction is still considered. FlashAttention is therefore not an approximate-attention method; up to ordinary floating-point evaluation order, it computes the same mathematical function as @eq-dense-attention while changing the schedule of reads, writes, and reductions @dao2022flashattention.

== Online Softmax <sec-online-softmax>

Tiling introduces a genuine normalization problem. A Softmax row depends on all its logits, but a query tile sees keys only one block at a time. Consider one row whose already processed logits form set $A$ and whose newly processed block forms set $C$. Define the partial maxima, normalization factors, and unnormalized value accumulators as

$
  m_A = max_(j in A) s_j, quad
  ell_A = sum_(j in A) exp(s_j - m_A), quad
  n_A = sum_(j in A) exp(s_j - m_A) v_j,
$ <eq-online-softmax-partial-state>

with analogous quantities $m_C$, $ell_C$, and $n_C$ for the new block. Let $m = op("max")(m_A, m_C)$. The combined state is

$
  ell = exp(m_A - m) ell_A + exp(m_C - m) ell_C,
  quad
  n = exp(m_A - m) n_A + exp(m_C - m) n_C,
  quad
  o = frac(n, ell).
$ <eq-online-softmax-merge>

The merge in @eq-online-softmax-merge is exact in real arithmetic. When a later block has a larger maximum, the earlier partial sum and value accumulator are rescaled to the new stable reference. The final ratio equals the ordinary Softmax-weighted value sum, but neither the global score row nor the global probability row needs to be stored. Max subtraction also controls exponential range, connecting this execution method to Chapter 8's discussion of numerically stable Softmax.

== Memory and Later Improvements <sec-flashattention-memory-and-improvements>

Dense attention's *mathematical interactions* remain quadratic. FlashAttention does not remove the $T^2$ score relationships or promise that every workload becomes compute-bound. Its key benefit is reducing the additional intermediate-memory footprint and HBM traffic associated with materializing scores and probabilities. Backward computation can recompute local quantities rather than retaining the full attention matrix, trading controlled arithmetic for less activation storage.

FlashAttention-2 retains this IO-aware exactness while improving work partitioning, parallelism, and non-matrix-multiplication overhead @dao2024flashattention2. The reusable lesson is not a version-specific kernel recipe. Execution performance depends on how work is divided across available processors and on whether the implementation makes productive use of the memory hierarchy. Shapes, masking form, dtype, dropout, and hardware determine whether a particular kernel path realizes a large gain; a familiar kernel name is not itself a performance result.

= Head-Representation Designs <sec-head-representation-designs>

Chapter 3 introduced attention with a general mapping from Query heads to KV heads. Here the central distinction is restated compactly. MHA assigns independent Key and Value heads to every Query head. MQA keeps many Query heads but shares one Key head and one Value head. GQA shares KV heads within groups of Query heads. These are not three execution algorithms for the same checkpoint: they are architectural configurations of the learned $K$ and $V$ projections.

== MHA, MQA, and GQA <sec-mha-mqa-gqa>

For ordinary Multi-Head Attention (MHA),

$
  H_"kv" = H_q.
$ <eq-mha-kv-head-count>

Each Query head has a distinct Key head and Value head. This is the standard multi-head form developed in Chapter 3. Multi-Query Attention (MQA) preserves $H_q$ Query heads but uses one shared Key and one shared Value head:

$
  H_"kv" = 1.
$ <eq-mqa-kv-head-count>

The sharing reduces cached state and repeated Decode-time KV reads, which motivated MQA as a faster Transformer decoding design @shazeer2019fast. Its cost is an aggressive restriction: all Query heads must address the same per-token Key/Value representation.

Grouped-Query Attention (GQA) supplies an intermediate choice:

$
  1 < H_"kv" < H_q.
$ <eq-gqa-kv-head-count>

Query heads are partitioned into groups, and every group shares one Key head and one Value head. GQA thereby retains more independent KV representations than MQA while reducing them relative to MHA. Ainslie et al. introduce this interpolation and show why it is a useful quality-efficiency design space rather than a binary MHA-versus-MQA decision @ainslie2023gqa.

== KV Cache Consequences <sec-head-design-kv-cache>

For dense decoder-only attention, the ideal persistent cache payload has the familiar form

$
  M_"KV" approx 2 B L T H_"kv" d_h b.
$ <eq-head-design-kv-memory>

The factor two accounts for Keys and Values. Decreasing $H_"kv"$ directly decreases cache capacity and cache-bandwidth demand even though $H_q$ can remain fixed. It does not reduce the number of Query rows or eliminate dense query-key interactions. This is why head sharing is particularly consequential during Decode, where each newly generated Query reads a growing history of cached Keys and Values. Chapter 24 gives the broader lifecycle, paging, sharing, and quantization consequences of this equation.

#figure(
  academic-table(
    columns: (1.0fr, 1.1fr, 1.55fr, 1.45fr),
    align: (left, center, left, left),
    header: (
      [*Design*], [*$H_"kv"$*], [*KV representation*], [*Primary Decode consequence*],
    ),
    rows: (
      [MHA], [$H_q$], [One independent KV head per Query head.], [Largest conventional KV Cache and KV-read volume.],
      [MQA], [$1$], [One shared KV head for all Query heads.], [Minimum head-count cache footprint; strongest sharing constraint.],
      [GQA], [$1 < H_"kv" < H_q$], [One KV head per Query-head group.], [Intermediate cache and bandwidth trade-off.],
    ),
  ),
  caption: [MHA, MQA, and GQA differ in the learned organization of Key/Value representations. The Query-head count need not change when the KV-head count is reduced.],
) <tab-head-sharing-comparison>

= Multi-Head Latent Attention <sec-multi-head-latent-attention>

Multi-Head Latent Attention (MLA) targets the same broad Decode bottleneck with a different compression mechanism. Instead of only reducing the number of full KV heads, it stores a lower-dimensional latent representation from which content-related Key and Value information is reconstructed or used by head-specific projections. MLA was introduced in the DeepSeek-V2 technical report as a KV-compression architecture @deepseek2024v2.

For an input hidden state $h_t$, a schematic latent mapping is

$
  c_t = W_c h_t, quad c_t in R^(d_c), quad d_c << 2 H_"kv" d_h.
$ <eq-mla-latent-state>

The inequality expresses the design objective, not a universal MLA definition: the latent width $d_c$ is chosen to be much smaller than storing independent full Key and Value vectors. Subsequent projections can derive attention-specific content representations from $c_t$. A conceptual form is

$
  k_t^"content" = W_K^"up" c_t, quad
  v_t = W_V^"up" c_t.
$ <eq-mla-up-projections>

The constructions in @eq-mla-latent-state and @eq-mla-up-projections convey the compression principle without claiming that every MLA implementation has identical factorization, ranks, or kernels. The serving cache can retain the latent state rather than all expanded KV vectors when the attention implementation is designed to use it. That changes both cache representation and the computation that reads or reconstructs it.

== Positional Components and RoPE <sec-mla-positional-components>

Content compression interacts delicately with position encoding. Chapter 3 explains that RoPE rotates query and key coordinates so their inner product carries relative displacement. A low-rank latent representation that is convenient for content projections need not preserve the same positional behavior after arbitrary reconstruction. MLA-style architectures can therefore keep a positional Key component separate from compressed content state and combine it with a matching positional Query component during score formation.

Conceptually, one may write

$
  q_t = [q_t^"content" ; q_t^"pos"], quad
  k_j = [k_j^"content" ; k_j^"pos"],
$ <eq-mla-decoupled-position>

where only the positional components receive the appropriate RoPE transformation. This is an architectural compatibility mechanism, not an instruction to split every Transformer head this way. Its importance is general: a cache-compression design must preserve the positional convention used by the checkpoint, rather than treating position information as an interchangeable post-processing detail.

== MLA Compared with Head Sharing <sec-mla-versus-head-sharing>

MQA and GQA reduce *how many* independently stored KV heads exist. MLA compresses *what each token stores* into a lower-dimensional latent state. Both can reduce KV Cache capacity and Decode bandwidth, but their quality, projection cost, kernel requirements, and compatibility constraints differ. A comparison should consequently state the representation, cache dtype, sequence length, batch regime, and serving kernels rather than ranking the names in isolation.

#figure(
  academic-table(
    columns: (0.85fr, 1.55fr, 1.35fr, 1.55fr),
    align: (left, left, left, left),
    header: (
      [*Design*], [*Stored per-token state*], [*Main compression mechanism*], [*Important engineering condition*],
    ),
    rows: (
      [MHA], [Full KV vectors for every head.], [None beyond ordinary head factorization.], [Conventional kernels and largest cache payload.],
      [MQA], [One shared full KV pair.], [Share KV across all Query heads.], [Quality must tolerate maximal sharing.],
      [GQA], [One full KV pair per group.], [Share KV within Query-head groups.], [Group count must match checkpoint configuration.],
      [MLA], [Compressed latent KV state, plus required positional state.], [Low-dimensional latent representation and reconstruction.], [Architecture-specific projections, RoPE handling, and kernel support.],
    ),
  ),
  caption: [Architectural KV-compression strategies. MLA is not merely GQA with fewer heads: it changes the representation retained for each token.],
) <tab-architectural-kv-comparison>

= Kernel Optimization and Architecture Are Complementary <sec-kernel-versus-architecture>

FlashAttention changes neither $H_"kv"$ nor the learned meaning of a Key or Value. It reduces intermediate attention traffic during Prefill and training by executing the same dense operation in a different order. MQA, GQA, and MLA alter the checkpoint's attention representation and reduce persistent KV state or its movement during Decode. The distinctions are summarized separately because a single table that calls one design "more efficient" would confuse different resources.

#figure(
  academic-table(
    columns: (1.25fr, 1.55fr, 1.65fr, 1.45fr),
    align: (left, left, left, left),
    header: (
      [*Execution choice*], [*Mathematical attention function*], [*Primary resource changed*], [*Primary phase*],
    ),
    rows: (
      [Conventional dense execution], [Dense attention], [May materialize large score/probability intermediates.], [Training and Prefill.],
      [FlashAttention-style execution], [Same dense attention], [HBM traffic and intermediate-memory footprint.], [Training and Prefill; supported Decode shapes.],
    ),
  ),
  caption: [FlashAttention is an execution choice, not a head-sharing architecture. A model with MHA, MQA, GQA, or MLA can require a separate decision about its attention kernel.],
) <tab-kernel-comparison>

During training, FlashAttention can reduce activation-memory pressure and improve attention-kernel utilization, while MQA, GQA, or MLA are fixed architectural choices learned with the model. During inference, FlashAttention may improve Prefill and compatible attention computations; MQA, GQA, and MLA primarily change persistent cache capacity and Decode-time state movement. End-to-end gains can still be limited by projections, model weights, scheduling, communication, or other system costs. Chapter 29's bottleneck-first method is therefore the correct way to evaluate a combined design.

= Failure Modes and Practical Limits <sec-efficient-attention-failure-modes>

The clean conceptual boundaries above make common errors easier to diagnose. FlashAttention can be unavailable for a dtype, mask layout, head dimension, hardware generation, or runtime path. Very short sequences can leave tile or launch overhead more visible than saved HBM traffic. An incorrect online Softmax merge can cause numerical error or a silent mismatch with a dense reference, especially when masking and reduced precision are involved. It is not enough to report that a kernel was selected; a test must compare its permitted outputs and gradients with an appropriate reference tolerance.

Head-sharing designs have a different risk profile. Treating $H_q$ as $H_"kv"$ misstates cache capacity for MQA and GQA. A checkpoint's head mapping, projection shapes, and positional convention cannot be changed at serving time by merely changing a configuration field. Aggressive KV sharing can reduce model quality, and a nominal cache reduction may not improve latency if weights, scheduler delay, or another operation dominates. MLA adds projection, representation, and positional-encoding complexity; an implementation that expands and stores its full reconstructed KV state would forfeit much of the intended cache benefit.

= Implementation Contracts <sec-efficient-attention-implementation-contracts>

An attention-kernel contract must state the score scaling, causal or padding-mask semantics, supported tensor layouts, dtype and accumulation precision, dropout behavior in training, and the numerical tolerance against a reference dense implementation. Tests should include variable sequence lengths, masked rows, boundary tiles, and long rows whose maximum occurs in a later Key block. They should verify both forward outputs and, when training is supported, gradients. Memory reports should separate persistent KV Cache from temporary attention workspace; otherwise FlashAttention's intermediate-memory benefit and a head-design cache benefit will be conflated.

An architectural cache contract must record $H_q$, $H_"kv"$, $d_h$, layer count, cache dtype, the exact model revision, and positional convention. Its expected ideal payload should agree with @eq-head-design-kv-memory before allocator rounding and metadata. For MQA and GQA, tests should verify the Query-to-KV head mapping and cache tensor shape. For MLA, they must additionally verify the latent projection, any reconstruction or absorbed-projection algebra, decoupled positional components, and equivalence to the model's specified attention rule. Benchmarking must state Prefill and Decode lengths, batch concurrency, kernel path, and quality checks; a byte-count reduction alone is not a serving result.

= Summary <sec-efficient-attention-summary>

Dense attention combines quadratic query-key interactions with an execution problem: a naive schedule can move large score and probability intermediates through HBM. FlashAttention preserves the attention function while using tiles and online Softmax to avoid materializing the full attention matrix in slow memory. It reduces intermediate-memory demand and IO, but it does not make dense attention's all-pairs arithmetic linear.

MHA, MQA, GQA, and MLA address a separate bottleneck: the Key/Value state stored and read across autoregressive steps. MQA and GQA share full KV heads at different granularities, whereas MLA compresses KV information into a latent representation and requires compatible positional handling. FlashAttention can be paired with each of these designs. Chapter 31 then considers the separate architectural choice of computing only a structured subset of attention edges. The useful systems question is consequently not which label is globally best, but whether the measured workload is constrained by attention IO, persistent KV state, computation, or another component of the serving path.

#pagebreak()
#heading(level: 1, numbering: none, outlined: false)[References]
#chapter-bibliography("/references/handbook.bib")
