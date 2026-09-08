# The Foundation Model Handbook

*A technical monograph on Foundation Models and LLM systems.*

This repository is a maintained Typst manuscript for readers preparing to understand, implement, and reason about modern Foundation Model systems. It develops the mathematical, architectural, training, alignment, inference, and systems foundations through independently readable chapters with shared notation, bibliography, and restrained academic typography. It is a technical handbook developed chapter by chapter, not a generated encyclopedia or a collection of disconnected notes.

## Version 1.0

Version 1.0 completes the handbook's core six-area learning roadmap: LLM Architecture, Pretraining, Post-training / Alignment, Parameter-Efficient Fine-Tuning, Inference Optimization, and LLM Systems. The supporting Foundations and Efficient Attention chapters establish prerequisites; the completed Retrieval-Augmented Generation sequence is a companion extension.

The release edition is available as the [complete handbook PDF](build/the-foundation-model-handbook-v1.0.pdf). Standalone PDFs remain available in the chapter list below for focused reading and review.

## Handbook structure

| Part | Completed chapters | Scope |
| --- | --- | --- |
| Foundations | 1 | The discrete model interface: tokenization and input representations. |
| Architecture | 2--4 | Decoder-only Transformer computation, attention and position encoding, FFNs, normalization, and residual connections. |
| Pretraining | 5--11, 38 | Language modeling, data, optimization, numerical stability, scaling, distributed execution, FSDP state management, and training diagnostics. |
| Post-training | 12--18, 37, 39 | SFT, preference data, reward modeling, RLHF/PPO, DPO, GRPO, reasoning RL, on-policy improvement, distributed RL systems, and post-training evaluation. |
| Inference and Serving | 19--25 | Autoregressive generation, memory optimization, batching, scheduling, acceleration, distributed execution, and performance design. |
| Retrieval-Augmented Generation (complete) | 26--33 | External knowledge, semantic and sparse retrieval, vector indexing, document segmentation, reranking, advanced retrieval architectures, context construction, grounding, provenance, and RAG evaluation. |
| Parameter-Efficient Fine-Tuning | 34 | LoRA, QLoRA, Adapters, Prefix Tuning, Prompt Tuning, and adaptation-state trade-offs. |
| Efficient Attention | 35--36 | FlashAttention, MHA/MQA/GQA/MLA, sparse and long-context attention, and KV-state design trade-offs. |

## Core six-area roadmap

The chapter numbers preserve publication history; this dependency-aware traversal is the recommended route through the completed core.

| Roadmap area | Primary chapters | Role in the path |
| --- | --- | --- |
| LLM Architecture | 1--4; 35--36 | Start with the discrete input interface and decoder-only Transformer. Chapters 35--36 extend the attention model with efficient execution, KV representations, and long-context connectivity. |
| Pretraining | 5--11; 38 | Move from the language-model objective and data distribution to optimization, numerical stability, scale, distributed execution, diagnostics, and practical FSDP state. |
| Post-training / Alignment | 12--18; 37; 39 | Proceed from SFT and preference learning to online alignment, evaluation, and the distributed RL system that executes those algorithms. |
| Parameter-Efficient Fine-Tuning | 34 | Read after Chapter 12 to compare frozen-base adaptation methods under the same SFT data and masking contracts. |
| Inference Optimization | 19--25 | Develop the Prefill/Decode execution model, cache management, quantization, scheduling, speculation, distributed inference, and end-to-end performance design. |
| LLM Systems | 10; 24--25; 38--39 | Revisit these chapters as the cross-cutting systems sequence: distributed training, FSDP state, distributed serving, performance engineering, and RL worker orchestration. |

For a strictly sequential first pass, read Chapters 1--12, then Chapter 34, Chapters 13--18, Chapter 37, and Chapter 39. Read Chapters 19--25 next; revisit Chapters 35--36 after the attention, KV Cache, and serving foundations are established. Chapters 26--33 form the completed Retrieval-Augmented Generation extension.

## Completed chapters

### Foundations

- [Chapter 1 — Tokenization and Input Representations](build/foundations/01-tokenization-and-input-representations.pdf)

### Architecture

- [Chapter 2 — Transformer Architecture](build/architecture/02-transformer-architecture.pdf)
- [Chapter 3 — Attention and Position Encoding](build/architecture/03-attention-and-position-encoding.pdf)
- [Chapter 4 — Feed-Forward Networks, Normalization, and Residual Connections](build/architecture/04-feed-forward-normalization-and-residual-connections.pdf)

### Pretraining

- [Chapter 5 — Pretraining Objective and Language Modeling](build/pretraining/05-pretraining-objective-and-language-modeling.pdf)
- [Chapter 6 — Pretraining Data](build/pretraining/06-pretraining-data.pdf)
- [Chapter 7 — Optimization for Pretraining](build/pretraining/07-optimization-for-pretraining.pdf)
- [Chapter 8 — Numerical Precision and Training Stability](build/pretraining/08-numerical-precision-and-training-stability.pdf)
- [Chapter 9 — Scaling Laws and Compute](build/pretraining/09-scaling-laws-and-compute.pdf)
- [Chapter 10 — Distributed Training](build/pretraining/10-distributed-training.pdf)
- [Chapter 11 — Evaluation, Checkpointing, and Training Diagnostics](build/pretraining/11-evaluation-checkpointing-and-training-diagnostics.pdf)
- [Chapter 38 — Practical FSDP and Distributed Training State](build/pretraining/38-practical-fsdp-and-distributed-training-state.pdf)

### Post-training

- [Chapter 12 — Supervised Fine-Tuning](build/post-training/12-supervised-fine-tuning.pdf)
- [Chapter 13 — Preference Data and Reward Modeling](build/post-training/13-preference-data-and-reward-modeling.pdf)
- [Chapter 14 — RLHF and PPO](build/post-training/14-rlhf-and-ppo.pdf)
- [Chapter 15 — Direct Preference Optimization](build/post-training/15-direct-preference-optimization.pdf)
- [Chapter 16 — Group Relative Policy Optimization](build/post-training/16-group-relative-policy-optimization.pdf)
- [Chapter 17 — Reasoning RL, Rollouts, and Verifiable Rewards](build/post-training/17-reasoning-rl-rollouts-and-verifiable-rewards.pdf)
- [Chapter 18 — Post-Training Evaluation and Alignment Trade-offs](build/post-training/18-post-training-evaluation-and-alignment-trade-offs.pdf)
- [Chapter 37 — On-Policy Alignment and Iterative Policy Improvement](build/post-training/37-on-policy-alignment-and-iterative-policy-improvement.pdf)
- [Chapter 39 — Distributed RL Training Systems](build/post-training/39-distributed-rl-training-systems.pdf)

### Parameter-Efficient Fine-Tuning

- [Chapter 34 — Parameter-Efficient Fine-Tuning](build/parameter-efficient-fine-tuning/34-parameter-efficient-fine-tuning.pdf)

### Efficient Attention

- [Chapter 35 — Efficient Attention and Head-Representation Design](build/efficient-attention/35-efficient-attention-and-head-representation-design.pdf)
- [Chapter 36 — Sparse and Long-Context Attention](build/efficient-attention/36-sparse-and-long-context-attention.pdf)

### Inference and Serving

- [Chapter 19 — LLM Inference Fundamentals](build/inference-serving/19-llm-inference-fundamentals.pdf)
- [Chapter 20 — KV Cache and Memory Optimization](build/inference-serving/20-kv-cache-and-memory-optimization.pdf)
- [Chapter 21 — Quantization for LLM Inference](build/inference-serving/21-quantization-for-llm-inference.pdf)
- [Chapter 22 — Batching, Scheduling, and LLM Serving Systems](build/inference-serving/22-batching-scheduling-and-llm-serving-systems.pdf)
- [Chapter 23 — Speculative Decoding and Inference Acceleration](build/inference-serving/23-speculative-decoding-and-inference-acceleration.pdf)
- [Chapter 24 — Distributed LLM Inference and Parallelism](build/inference-serving/24-distributed-llm-inference-and-parallelism.pdf)
- [Chapter 25 — Inference System Design and Performance Optimization](build/inference-serving/25-inference-system-design-and-performance-optimization.pdf)

### Retrieval-Augmented Generation

- [Chapter 26 — Retrieval-Augmented Generation Fundamentals](build/rag-knowledge-augmentation/26-retrieval-augmented-generation-fundamentals.pdf)
- [Chapter 27 — Embeddings and Semantic Retrieval](build/rag-knowledge-augmentation/27-embeddings-and-semantic-retrieval.pdf)
- [Chapter 28 — Vector Search and Approximate Nearest Neighbors](build/rag-knowledge-augmentation/28-vector-search-and-approximate-nearest-neighbors.pdf)
- [Chapter 29 — Chunking and Document Segmentation](build/rag-knowledge-augmentation/29-chunking-and-document-segmentation.pdf)
- [Chapter 30 — Sparse Retrieval and Hybrid Search](build/rag-knowledge-augmentation/30-sparse-retrieval-and-hybrid-search.pdf)
- [Chapter 31 — Reranking and Retrieval Refinement](build/rag-knowledge-augmentation/31-reranking-and-retrieval-refinement.pdf)
- [Chapter 32 — Advanced Retrieval and RAG Architectures](build/rag-knowledge-augmentation/32-advanced-retrieval-and-rag-architectures.pdf)
- [Chapter 33 — RAG Evaluation and Diagnostics](build/rag-knowledge-augmentation/33-rag-evaluation-and-diagnostics.pdf)

Future editions may add peer sections only when their first chapter is ready for the same source, layout, and review standard. Those possible extensions are outside the v1.0 core scope.

## Repository layout

```text
main.typ                      Complete v1.0 handbook source, in intended reading order
chapters/
  foundations/                Chapter 1: tokenization and input representations
  architecture/               Chapters 2--4: Transformer architecture
  pretraining/                Chapters 5--11 and 38: objective, data, optimization, systems, and diagnostics
  post-training/              Chapters 12--18, 37, and 39: SFT, preference learning, RL, and RL systems
  inference-serving/          Chapters 19--25: execution, cache, scheduling, acceleration, and serving systems
  parameter-efficient-fine-tuning/ Chapter 34: low-rank, quantized-base, adapter, prefix, and prompt adaptation
  efficient-attention/        Chapters 35--36: FlashAttention, KV representation, and sparse long-context attention
  rag-knowledge-augmentation/ Chapters 26--33: retrieval, context construction, advanced architectures, and evaluation
templates/
  typst/                      Shared book and chapter layout, environments, notation, and typography
references/
  handbook.bib                Shared BibTeX database
assets/
  figures/                    Reusable figure assets
  tables/                     Reusable table data and assets
docs/
  book-outline.md             Topic roadmap and published-chapter map
  style-guide.md              Writing, notation, and visual conventions
  chapter-template.md         Adaptable chapter structure
  codex-instructions.md       Required authoring and review workflow
scripts/
  build-release.sh            Rebuild the complete v1.0 PDF
  build-all-chapters.sh       Rebuild every standalone chapter PDF
build/
  the-foundation-model-handbook-v1.0.pdf  Complete release edition
  <part>/                     Versioned standalone chapter PDFs
```

Each chapter resides in `chapters/<part>/<number>-<topic>/main.typ`. Shared assets belong in `assets/`; a chapter-specific asset directory may be added inside its chapter only when that asset is not reused elsewhere.

## Build the release edition

Install [Typst](https://typst.app/) and build the complete v1.0 handbook:

```bash
./scripts/build-release.sh
```

The command compiles `main.typ` in the intended reading order and writes `build/the-foundation-model-handbook-v1.0.pdf`. It passes the `handbook=true` build input internally so that standalone chapter sources retain their local title-page presentation while the assembled edition uses book-level front matter, parts, table of contents, and running headers.

## Build standalone chapters

Build every published standalone chapter:

```bash
./scripts/build-all-chapters.sh
```

To build one chapter independently, preserve its part and slug in the output path:

```bash
typst compile --root . chapters/pretraining/09-scaling-laws-and-compute/main.typ build/pretraining/09-scaling-laws-and-compute.pdf
```

The thirty-nine standalone PDFs are versioned so that their layout and writing style can be reviewed directly from the repository. The build helper discovers chapter `main.typ` files automatically, so a new peer section joins the standalone build without duplicating a command list.

## References and authoring

All chapters draw from the shared [BibTeX bibliography](references/handbook.bib). Entries favor DOI-backed source records and stable primary or standard references; each standalone chapter prints the references it uses so it can be read independently, and the complete edition preserves those local reference lists.

The handbook's authoring contract is documented in the [style guide](docs/style-guide.md), [adaptable chapter template](docs/chapter-template.md), and [chapter workflow](docs/codex-instructions.md). Any future contribution should preserve the shared Typst infrastructure, compile both the affected standalone chapter and the assembled release edition when applicable, and visually inspect the resulting PDFs before publication.

## Development principles

- Treat each chapter as a self-contained technical document, not a blog post or a set of interview notes.
- Prefer source-backed explanation, explicit notation, and complete local references over broad but unsupported coverage.
- Keep shared template code, notation conventions, bibliography infrastructure, and reusable assets centralized.
- Add equations, figures, tables, and formal environments only when they improve the exposition.
- Publish a later topic only after its technical claims and chapter-level presentation have been checked.
