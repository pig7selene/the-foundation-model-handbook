#let definition-counter = counter("definition")
#let theorem-counter = counter("theorem")
#let proposition-counter = counter("proposition")
#let lemma-counter = counter("lemma")
#let remark-counter = counter("remark")
#let example-counter = counter("example")
#let algorithm-counter = counter("algorithm")

#let environment(label, counter, title: none, body) = {
  counter.step()
  block(
    width: 100%,
    inset: (left: 1em, right: 0.8em, top: 0.5em, bottom: 0.5em),
    above: 0.8em,
    below: 0.8em,
    stroke: (left: 0.8pt + black),
  )[
    #text(weight: "semibold")[#label #context counter.display("1")#if title != none [: #title].]
    #body
  ]
}

#let definition(title: none, body) = environment([Definition], definition-counter, title: title, body)
#let theorem(title: none, body) = environment([Theorem], theorem-counter, title: title, body)
#let proposition(title: none, body) = environment([Proposition], proposition-counter, title: title, body)
#let lemma(title: none, body) = environment([Lemma], lemma-counter, title: title, body)
#let remark(title: none, body) = environment([Remark], remark-counter, title: title, body)
#let example(title: none, body) = environment([Example], example-counter, title: title, body)
#let algorithm(title: none, body) = environment([Algorithm], algorithm-counter, title: title, body)

// Typst's built-in `figure`, `table`, and `equation` elements provide captions,
// automatic numbering, labels, and references. Use `<label>` immediately after
// an element, then cite it with `@label`.
