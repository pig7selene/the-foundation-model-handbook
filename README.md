# Foundation Model Technical Notes

*Rigorous standalone chapters for Foundation Model and LLM systems study.*

This repository is a growing collection of independent Typst chapters for the core knowledge needed in Foundation Model and LLM algorithm work. Each PDF is designed to stand on its own while sharing a common technical vocabulary, mathematical notation, bibliography, and restrained academic visual language.

## Published chapters

### Architecture

- [Chapter 1 — Tokenization and Input Representations](build/01-tokenization-and-input-representations.pdf)
- [Chapter 2 — Transformer Architecture](build/02-transformer-architecture.pdf)
- [Chapter 3 — Attention and Position Encoding](build/03-attention-and-position-encoding.pdf)
- [Chapter 4 — Feed-Forward Networks, Normalization, and Residual Connections](build/04-feed-forward-normalization-and-residual-connections.pdf)

### Training

- [Chapter 5 — Pretraining Objective and Language Modeling](build/05-pretraining-objective-and-language-modeling.pdf)
- [Chapter 6 — Pretraining Data](build/06-pretraining-data.pdf)
- [Chapter 7 — Optimization for Pretraining](build/07-optimization-for-pretraining.pdf)

### Post-training

TODO

### Inference

TODO

### Systems

TODO

## Project structure

```text
template/                Shared standalone-chapter layout, environments, notation, and typography
chapters/01-tokenization/                Chapter 1 source and chapter-local figures
chapters/02-transformer-architecture/    Chapter 2 source and chapter-local figures
chapters/03-attention-and-position-encoding/  Chapter 3 source and chapter-local figures
chapters/04-feed-forward-normalization-and-residual-connections/  Chapter 4 source and chapter-local figures
chapters/05-pretraining-objective-and-language-modeling/  Chapter 5 source and chapter-local figures
chapters/06-pretraining-data/               Chapter 6 source and chapter-local figures
chapters/07-optimization-for-pretraining/    Chapter 7 source and chapter-local figures
figures/                 Shared vector figures, when a figure is reused
references/              Shared BibTeX database
scripts/                 Maintenance helpers
build/                   Published chapter PDFs
BOOK_OUTLINE.md          Topic roadmap for future chapters
```

## Build the published chapters

Install [Typst](https://typst.app/) and compile each chapter independently:

```bash
typst compile --root . chapters/01-tokenization/main.typ build/01-tokenization-and-input-representations.pdf
typst compile --root . chapters/02-transformer-architecture/main.typ build/02-transformer-architecture.pdf
typst compile --root . chapters/03-attention-and-position-encoding/main.typ build/03-attention-and-position-encoding.pdf
typst compile --root . chapters/04-feed-forward-normalization-and-residual-connections/main.typ build/04-feed-forward-normalization-and-residual-connections.pdf
typst compile --root . chapters/05-pretraining-objective-and-language-modeling/main.typ build/05-pretraining-objective-and-language-modeling.pdf
typst compile --root . chapters/06-pretraining-data/main.typ build/06-pretraining-data.pdf
typst compile --root . chapters/07-optimization-for-pretraining/main.typ build/07-optimization-for-pretraining.pdf
```

The seven current PDFs are versioned so that their layout and writing style can be reviewed directly from the repository.

## Development principles

- Treat each chapter as a self-contained technical document, not a blog post or a set of interview notes.
- Prefer source-backed explanation, explicit notation, and complete local references over broad but unsupported coverage.
- Keep shared template code, notation conventions, and bibliography infrastructure centralized.
- Add equations, figures, tables, and formal environments only when they improve the exposition.
- Publish a later topic only after its technical claims and chapter-level presentation have been checked.
