# The Foundation Model Handbook

*A technical monograph on Foundation Models and LLM systems.*

This repository is a maintained Typst manuscript for readers who want to understand, implement, and reason about modern Foundation Model systems. It develops mathematical, architectural, training, alignment, inference, and systems foundations through independently readable chapters with shared notation, bibliography, and restrained academic typography. It is a technical handbook developed chapter by chapter—not a generated encyclopedia or a collection of disconnected notes.

## Version 1.0

Version 1.0 completes the core roadmap: LLM Architecture, Pretraining, Post-training / Alignment, Parameter-Efficient Fine-Tuning, Inference Optimization, and LLM Systems. Foundations, Efficient Attention, and Retrieval-Augmented Generation provide the prerequisite and extension material needed to connect those areas into a coherent reading path.

The release edition is available as the [complete handbook PDF](build/the-foundation-model-handbook-v1.0.pdf). Focused standalone PDFs are linked below.

## Reading order

The chapter sequence is the intended reading order. It is strictly monotonic from Chapter 1 through Chapter 39.

| Part | Chapters | Scope |
| --- | --- | --- |
| I. Foundations | 1 | Tokenization and the discrete model interface. |
| II. Architecture | 2--4 | Decoder-only Transformer computation, attention and position encoding, FFNs, normalization, and residual connections. |
| III. Pretraining | 5--12 | Objective, data, optimization, numerical stability, scaling, distributed training, diagnostics, and practical FSDP state. |
| IV. Post-training, Alignment, and Adaptation | 13--22 | SFT, PEFT, preference learning, online alignment, evaluation, and distributed RL systems. |
| V. Inference and Serving | 23--29 | Autoregressive execution, caches, quantization, batching, speculation, distributed inference, and end-to-end optimization. |
| VI. Efficient Attention and Long Context | 30--31 | FlashAttention, KV-representation designs, sparse attention, and long-context trade-offs. |
| VII. Retrieval-Augmented Generation and Knowledge Augmentation | 32--39 | Retrieval, indexing, chunking, hybrid search, reranking, advanced RAG, and evaluation. |

## Core six-area roadmap

| Roadmap area | Primary chapters | Role |
| --- | --- | --- |
| LLM Architecture | 1--4, 30--31 | Establishes the token interface and Transformer, then extends attention execution and long-context design. |
| Pretraining | 5--12 | Moves from language modeling and data to optimization, stability, scaling, distributed execution, diagnostics, and FSDP state. |
| Post-training / Alignment | 13, 15--22 | Covers SFT, preference learning, RLHF, DPO, GRPO, reasoning RL, evaluation, online improvement, and distributed RL. |
| Parameter-Efficient Fine-Tuning | 14 | Compares LoRA, QLoRA, Adapters, Prefix Tuning, and Prompt Tuning under SFT contracts. |
| Inference Optimization | 23--31 | Develops Prefill/Decode, cache management, quantization, scheduling, speculation, distributed serving, and attention efficiency. |
| LLM Systems | 10--12, 22, 28--29 | Connects distributed training and FSDP state to RL worker orchestration and distributed serving. |

## Chapters

### Part I — Foundations

- [Chapter 1 — Tokenization and Input Representations](build/foundations/01-tokenization-and-input-representations.pdf)

### Part II — Architecture

- [Chapter 2 — Transformer Architecture](build/architecture/02-transformer-architecture.pdf)
- [Chapter 3 — Attention and Position Encoding](build/architecture/03-attention-and-position-encoding.pdf)
- [Chapter 4 — Feed-Forward Networks, Normalization, and Residual Connections](build/architecture/04-feed-forward-normalization-and-residual-connections.pdf)

### Part III — Pretraining

- [Chapter 5 — Pretraining Objective and Language Modeling](build/pretraining/05-pretraining-objective-and-language-modeling.pdf)
- [Chapter 6 — Pretraining Data](build/pretraining/06-pretraining-data.pdf)
- [Chapter 7 — Optimization for Pretraining](build/pretraining/07-optimization-for-pretraining.pdf)
- [Chapter 8 — Numerical Precision and Training Stability](build/pretraining/08-numerical-precision-and-training-stability.pdf)
- [Chapter 9 — Scaling Laws and Compute](build/pretraining/09-scaling-laws-and-compute.pdf)
- [Chapter 10 — Distributed Training](build/pretraining/10-distributed-training.pdf)
- [Chapter 11 — Evaluation, Checkpointing, and Training Diagnostics](build/pretraining/11-evaluation-checkpointing-and-training-diagnostics.pdf)
- [Chapter 12 — Practical FSDP and Distributed Training State](build/pretraining/12-practical-fsdp-and-distributed-training-state.pdf)

### Part IV — Post-training, Alignment, and Adaptation

- [Chapter 13 — Supervised Fine-Tuning](build/post-training/13-supervised-fine-tuning.pdf)
- [Chapter 14 — Parameter-Efficient Fine-Tuning](build/parameter-efficient-fine-tuning/14-parameter-efficient-fine-tuning.pdf)
- [Chapter 15 — Preference Data and Reward Modeling](build/post-training/15-preference-data-and-reward-modeling.pdf)
- [Chapter 16 — RLHF and PPO](build/post-training/16-rlhf-and-ppo.pdf)
- [Chapter 17 — Direct Preference Optimization](build/post-training/17-direct-preference-optimization.pdf)
- [Chapter 18 — Group Relative Policy Optimization](build/post-training/18-group-relative-policy-optimization.pdf)
- [Chapter 19 — Reasoning RL, Rollouts, and Verifiable Rewards](build/post-training/19-reasoning-rl-rollouts-and-verifiable-rewards.pdf)
- [Chapter 20 — Post-Training Evaluation and Alignment Trade-offs](build/post-training/20-post-training-evaluation-and-alignment-trade-offs.pdf)
- [Chapter 21 — On-Policy Alignment and Iterative Policy Improvement](build/post-training/21-on-policy-alignment-and-iterative-policy-improvement.pdf)
- [Chapter 22 — Distributed RL Training Systems](build/post-training/22-distributed-rl-training-systems.pdf)

### Part V — Inference and Serving

- [Chapter 23 — LLM Inference Fundamentals](build/inference-serving/23-llm-inference-fundamentals.pdf)
- [Chapter 24 — KV Cache and Memory Optimization](build/inference-serving/24-kv-cache-and-memory-optimization.pdf)
- [Chapter 25 — Quantization for LLM Inference](build/inference-serving/25-quantization-for-llm-inference.pdf)
- [Chapter 26 — Batching, Scheduling, and LLM Serving Systems](build/inference-serving/26-batching-scheduling-and-llm-serving-systems.pdf)
- [Chapter 27 — Speculative Decoding and Inference Acceleration](build/inference-serving/27-speculative-decoding-and-inference-acceleration.pdf)
- [Chapter 28 — Distributed LLM Inference and Parallelism](build/inference-serving/28-distributed-llm-inference-and-parallelism.pdf)
- [Chapter 29 — Inference System Design and Performance Optimization](build/inference-serving/29-inference-system-design-and-performance-optimization.pdf)

### Part VI — Efficient Attention and Long Context

- [Chapter 30 — Efficient Attention and Head-Representation Design](build/efficient-attention/30-efficient-attention-and-head-representation-design.pdf)
- [Chapter 31 — Sparse and Long-Context Attention](build/efficient-attention/31-sparse-and-long-context-attention.pdf)

### Part VII — Retrieval-Augmented Generation and Knowledge Augmentation

- [Chapter 32 — Retrieval-Augmented Generation Fundamentals](build/rag-knowledge-augmentation/32-retrieval-augmented-generation-fundamentals.pdf)
- [Chapter 33 — Embeddings and Semantic Retrieval](build/rag-knowledge-augmentation/33-embeddings-and-semantic-retrieval.pdf)
- [Chapter 34 — Vector Search and Approximate Nearest Neighbors](build/rag-knowledge-augmentation/34-vector-search-and-approximate-nearest-neighbors.pdf)
- [Chapter 35 — Chunking and Document Segmentation](build/rag-knowledge-augmentation/35-chunking-and-document-segmentation.pdf)
- [Chapter 36 — Sparse Retrieval and Hybrid Search](build/rag-knowledge-augmentation/36-sparse-retrieval-and-hybrid-search.pdf)
- [Chapter 37 — Reranking and Retrieval Refinement](build/rag-knowledge-augmentation/37-reranking-and-retrieval-refinement.pdf)
- [Chapter 38 — Advanced Retrieval and RAG Architectures](build/rag-knowledge-augmentation/38-advanced-retrieval-and-rag-architectures.pdf)
- [Chapter 39 — RAG Evaluation and Diagnostics](build/rag-knowledge-augmentation/39-rag-evaluation-and-diagnostics.pdf)

## Repository layout

```text
main.typ                      Complete v1.0 handbook source, in reading order
chapters/
  foundations/                Chapter 1
  architecture/               Chapters 2--4
  pretraining/                Chapters 5--12
  post-training/              Chapters 13, 15--22
  parameter-efficient-fine-tuning/ Chapter 14
  inference-serving/          Chapters 23--29
  efficient-attention/        Chapters 30--31
  rag-knowledge-augmentation/ Chapters 32--39
templates/typst/              Shared book and chapter layout, notation, and typography
references/handbook.bib       Shared BibTeX database
assets/                       Reusable figures and tables
docs/                         Roadmap, style guide, chapter template, and authoring workflow
scripts/                      Standalone and release build helpers
build/                        Versioned standalone PDFs and the assembled release PDF
```

Each chapter is stored at `chapters/<part>/<number>-<topic>/main.typ`. Shared material stays centralized in `templates/typst/`, `references/`, and `assets/`.

## Build

Install [Typst](https://typst.app/), then build the complete release edition:

```bash
./scripts/build-release.sh
```

This compiles `main.typ` with the book-level front matter, table of contents, parts, running headers, and page numbering, writing `build/the-foundation-model-handbook-v1.0.pdf`.

Build every standalone chapter:

```bash
./scripts/build-all-chapters.sh
```

To compile one chapter:

```bash
typst compile --root . chapters/pretraining/09-scaling-laws-and-compute/main.typ build/pretraining/09-scaling-laws-and-compute.pdf
```

## References and authoring

All chapters draw from the shared [BibTeX bibliography](references/handbook.bib). Entries favor DOI-backed primary or standard references. A standalone chapter prints only the works it cites, while the assembled book preserves those local reference lists.

The [style guide](docs/style-guide.md), [adaptable chapter template](docs/chapter-template.md), and [authoring workflow](docs/codex-instructions.md) document the project contract. Future work must preserve the shared Typst infrastructure, compile affected standalone chapters and the complete book where appropriate, and visually inspect the resulting PDFs before publication.

## Development principles

- Treat each chapter as a self-contained technical document, not a blog post or interview note.
- Prefer source-backed explanation, explicit notation, and complete local references over broad but unsupported coverage.
- Keep shared template code, notation conventions, bibliography infrastructure, and reusable assets centralized.
- Add equations, figures, tables, and formal environments only when they improve the exposition.
- Extend the handbook only when a new topic meets the same source, layout, and review standard; future subjects are outside the v1.0 core scope.
