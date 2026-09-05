# The Foundation Model Handbook

*Rigorous standalone chapters for Foundation Model and LLM systems study.*

This repository is a long-term collection of independently compiled Typst chapters. It develops the mathematical, architectural, and systems foundations needed for Foundation Model work while maintaining one shared notation system, bibliography, and restrained academic visual language. The manuscript is organized into Foundations, Architecture, Pretraining, and Post-training, with later parts added only when their chapters are ready.

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

Later parts, such as Reinforcement Learning, Inference and Serving, and RAG / Agents / LLM Applications, will be added as peer directories under `chapters/` and `build/` when their first chapters are ready.

## Repository layout

```text
chapters/
  foundations/                 Chapter 1: tokenization and input representations
  architecture/                Chapters 2–4: Transformer architecture
  pretraining/                 Chapters 5–11: pretraining objectives, data, and systems
  post-training/               Chapter 12 onward: supervised and preference-based adaptation
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

Install [Typst](https://typst.app/) and compile each chapter independently:

```bash
typst compile --root . chapters/foundations/01-tokenization-and-input-representations/main.typ build/foundations/01-tokenization-and-input-representations.pdf
typst compile --root . chapters/architecture/02-transformer-architecture/main.typ build/architecture/02-transformer-architecture.pdf
typst compile --root . chapters/architecture/03-attention-and-position-encoding/main.typ build/architecture/03-attention-and-position-encoding.pdf
typst compile --root . chapters/architecture/04-feed-forward-normalization-and-residual-connections/main.typ build/architecture/04-feed-forward-normalization-and-residual-connections.pdf
typst compile --root . chapters/pretraining/05-pretraining-objective-and-language-modeling/main.typ build/pretraining/05-pretraining-objective-and-language-modeling.pdf
typst compile --root . chapters/pretraining/06-pretraining-data/main.typ build/pretraining/06-pretraining-data.pdf
typst compile --root . chapters/pretraining/07-optimization-for-pretraining/main.typ build/pretraining/07-optimization-for-pretraining.pdf
typst compile --root . chapters/pretraining/08-numerical-precision-and-training-stability/main.typ build/pretraining/08-numerical-precision-and-training-stability.pdf
typst compile --root . chapters/pretraining/09-scaling-laws-and-compute/main.typ build/pretraining/09-scaling-laws-and-compute.pdf
typst compile --root . chapters/pretraining/10-distributed-training/main.typ build/pretraining/10-distributed-training.pdf
typst compile --root . chapters/pretraining/11-evaluation-checkpointing-and-training-diagnostics/main.typ build/pretraining/11-evaluation-checkpointing-and-training-diagnostics.pdf
typst compile --root . chapters/post-training/12-supervised-fine-tuning/main.typ build/post-training/12-supervised-fine-tuning.pdf
typst compile --root . chapters/post-training/13-preference-data-and-reward-modeling/main.typ build/post-training/13-preference-data-and-reward-modeling.pdf
typst compile --root . chapters/post-training/14-rlhf-and-ppo/main.typ build/post-training/14-rlhf-and-ppo.pdf
```

The fourteen current PDFs are versioned so that their layout and writing style can be reviewed directly from the repository.

## Development principles

- Treat each chapter as a self-contained technical document, not a blog post or a set of interview notes.
- Prefer source-backed explanation, explicit notation, and complete local references over broad but unsupported coverage.
- Keep shared template code, notation conventions, bibliography infrastructure, and reusable assets centralized.
- Add equations, figures, tables, and formal environments only when they improve the exposition.
- Publish a later topic only after its technical claims and chapter-level presentation have been checked.
