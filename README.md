# The Foundation Model Handbook

*Architecture, Pretraining, Post-training, Reinforcement Learning, Inference, and Systems*

`The Foundation Model Handbook` is a long-lived personal technical handbook for the core knowledge needed in foundation-model algorithm work. Its destination is a carefully edited PDF with the character of an academic textbook or technical monograph, rather than a collection of Markdown notes.

## Goal

The handbook records understanding accumulated through actual study, implementation, and reading. It is deliberately not an automatically generated AI encyclopedia: material that has not yet been studied or verified remains a TODO.

## Current status

The repository currently contains the book architecture, Typst template, bibliography plumbing, and chapter placeholders. It intentionally does **not** yet contain substantive textbook chapters.

The canonical 44-chapter writing plan, including learning objectives, prerequisites, expected depth, and the dependency order, is in [BOOK_OUTLINE.md](BOOK_OUTLINE.md).

## Layout

```text
main.typ                 Book entry point
chapters/                Eight parts, each with an index.typ skeleton
template/                Book, environments, notation, and typography definitions
figures/                 Curated figures for the book
tables/                  Table data sources for the book
references/              BibTeX database
assets/                  Supporting non-figure assets
sources/                 Reading notes and source materials awaiting synthesis
scripts/                 Book maintenance helpers
build/                   Local compilation output (ignored except .gitkeep)
```

## Build

Install [Typst](https://typst.app/) and run:

```bash
typst compile main.typ build/the-foundation-model-handbook.pdf
```

The generated PDF is intentionally excluded from version control.

## Development principles

- Prefer precise, source-backed explanations to broad coverage.
- Preserve standard technical terms such as Transformer, RLHF, KV Cache, and FSDP.
- Add equations, figures, examples, and references only when they clarify a studied concept.
- Keep unfinished material explicit with TODO markers rather than filling gaps with unverified prose.
- Maintain a restrained academic layout suitable for sustained reading and eventual print-quality PDF output.
