#let body-font = ("Libertinus Serif", "New Computer Modern", "Times New Roman")

#let setup-typography() = {
  set text(font: body-font, size: 10.5pt, lang: "en")
  set par(justify: true, leading: 0.68em, first-line-indent: 1.4em)
  set heading(numbering: "1.1")
  set heading.where(level: 1, outlined: true)(
    numbering: "I",
    supplement: [Part],
  )
  set heading.where(level: 2, outlined: true)(
    supplement: [Chapter],
  )

  show heading.where(level: 1): it => [
    #pagebreak(weak: true)
    #v(2.4em)
    #text(size: 11pt, weight: "semibold", tracking: 0.08em)[PART #counter(heading).at(it.location()).display("I")]
    #v(0.7em)
    #text(size: 24pt, weight: "semibold")[#it.body]
    #v(1.5em)
  ]

  show heading.where(level: 2): it => [
    #v(1.6em)
    #text(size: 15pt, weight: "semibold")[#counter(heading).at(it.location()).display("1.1") #it.body]
    #v(0.4em)
  ]

  show heading.where(level: 3): it => [
    #v(1em)
    #text(size: 11pt, weight: "semibold")[#it.body]
    #v(0.25em)
  ]
}
