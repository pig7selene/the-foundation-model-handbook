# The Foundation Model Handbook

*Rigorous standalone chapters for Foundation Model and LLM systems study.*

This repository is a long-term collection of independently compiled Typst chapters. It develops the mathematical, architectural, and systems foundations needed for Foundation Model work while maintaining one shared notation system, bibliography, and restrained academic visual language. The manuscript is organized into Foundations, Architecture, Pretraining, and Post-training, with later parts added only when their chapters are ready.

## Handbook structure

| Part | Completed chapters | Scope |
| --- | --- | --- |
| Foundations | 1 | The discrete model interface: tokenization and input representations. |
| Architecture | 2--4 | Decoder-only Transformer computation, attention and position encoding, FFNs, normalization, and residual connections. |
| Pretraining | 5--11 | Language modeling, data, optimization, numerical stability, scaling, distributed execution, and training diagnostics. |
| Post-training | 12--18 | SFT, preference data, reward modeling, RLHF/PPO, DPO, GRPO, reasoning RL, and post-training evaluation. |

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

### Post-training

- [Chapter 12 — Supervised Fine-Tuning](build/post-training/12-supervised-fine-tuning.pdf)
- [Chapter 13 — Preference Data and Reward Modeling](build/post-training/13-preference-data-and-reward-modeling.pdf)
- [Chapter 14 — RLHF and PPO](build/post-training/14-rlhf-and-ppo.pdf)
- [Chapter 15 — Direct Preference Optimization](build/post-training/15-direct-preference-optimization.pdf)
- [Chapter 16 — Group Relative Policy Optimization](build/post-training/16-group-relative-policy-optimization.pdf)
- [Chapter 17 — Reasoning RL, Rollouts, and Verifiable Rewards](build/post-training/17-reasoning-rl-rollouts-and-verifiable-rewards.pdf)
- [Chapter 18 — Post-Training Evaluation and Alignment Trade-offs](build/post-training/18-post-training-evaluation-and-alignment-trade-offs.pdf)

Future peer parts will be added only when their first chapter is ready: Inference and Serving, Retrieval-Augmented Generation, Agents and Tool Use, and Multimodal Models. Each will use the same `chapters/<part>/` and `build/<part>/` layout, shared Typst infrastructure, bibliography, and review workflow.

## Repository layout

```text
chapters/
  foundations/                 Chapter 1: tokenization and input representations
  architecture/                Chapters 2–4: Transformer architecture
  pretraining/                 Chapters 5–11: pretraining objectives, data, and systems
  post-training/               Chapters 12--18: supervised and preference-based adaptation
templates/
  typst/                       Shared chapter layout, environments, notation, and typography
references/
  handbook.bib                 Shared BibTeX database
assets/
  figures/                     Reusable figure assets
  tables/                      Reusable table data and assets
docs/
  book-outline.md              Topic roadmap and published-chapter map
  style-guide.md               Writing, notation, and visual conventions
  chapter-template.md          Adaptable chapter structure
  codex-instructions.md        Required authoring and review workflow
scripts/                       Maintenance helpers
build/                         Versioned standalone chapter PDFs, grouped like `chapters/`
```

Each chapter resides in `chapters/<part>/<number>-<topic>/main.typ`. Shared assets belong in `assets/`; a chapter-specific asset directory may be added inside its chapter only when that asset is not reused elsewhere.

## Build the published chapters

Install [Typst](https://typst.app/) and build every published standalone chapter:

```bash
./scripts/build-all-chapters.sh
```

To build one chapter independently, preserve its part and slug in the output path:

```bash
typst compile --root . chapters/pretraining/09-scaling-laws-and-compute/main.typ build/pretraining/09-scaling-laws-and-compute.pdf
```

The eighteen current PDFs are versioned so that their layout and writing style can be reviewed directly from the repository. The build helper discovers `main.typ` files automatically, so a new peer section joins the complete build without duplicating a command list.

## Development principles

- Treat each chapter as a self-contained technical document, not a blog post or a set of interview notes.
- Prefer source-backed explanation, explicit notation, and complete local references over broad but unsupported coverage.
- Keep shared template code, notation conventions, bibliography infrastructure, and reusable assets centralized.
- Add equations, figures, tables, and formal environments only when they improve the exposition.
- Publish a later topic only after its technical claims and chapter-level presentation have been checked.
