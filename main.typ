#import "template/book.typ": handbook
#import "template/environments.typ": definition, theorem, proposition, lemma, remark, example, algorithm

#show: handbook.with(
  title: [The Foundation Model Handbook],
  subtitle: [Architecture, Pretraining, Post-training, Reinforcement Learning, Inference, and Systems],
  author: [Personal Technical Handbook],
)

= Preface

This is a living handbook for studying Foundation Models. Each chapter will be added only after the underlying ideas have been learned, checked against primary sources, and organized into a durable explanation.

#pagebreak()

#outline(title: [Contents], depth: 3)

#pagebreak()

#include "chapters/01-foundations/index.typ"
#include "chapters/02-pretraining/index.typ"
#include "chapters/03-post-training/index.typ"
#include "chapters/04-fine-tuning/index.typ"
#include "chapters/05-inference/index.typ"
#include "chapters/06-systems/index.typ"
#include "chapters/07-evaluation/index.typ"
#include "chapters/08-applications/index.typ"

#pagebreak()

#bibliography("references/references.bib", title: [References], full: true)
