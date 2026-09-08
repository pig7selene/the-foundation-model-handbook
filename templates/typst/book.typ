#import "typography.typ": body-font

#let book-section-numbering(..numbers) = {
  if numbers.len() == 3 {
    [#numbers.at(1).#numbers.at(2)]
  } else if numbers.len() >= 4 {
    [#numbers.at(1).#numbers.at(2).#numbers.at(3)]
  } else {
    numbering("1.1", ..numbers)
  }
}

#let book-header = context {
  let page-number = counter(page).get().first()
  if page-number <= 1 {
    []
  } else if calc.rem(page-number, 2) == 0 {
    align(left)[#text(size: 8.2pt, fill: luma(45%))[The Foundation Model Handbook]]
  } else {
    let chapters = query(heading.where(level: 2).before(here()))
    let title = if chapters.len() > 0 { chapters.last().body } else { [Contents] }
    align(right)[#text(size: 8.2pt, fill: luma(45%))[#title]]
  }
}

#let book-footer = context {
  align(center)[#text(size: 8.7pt)[#counter(page).display("1")]]
}

#let handbook-book(body) = {
  set document(
    title: [The Foundation Model Handbook],
    author: ("pig7selene",),
  )
  set text(font: body-font, size: 10.5pt, lang: "en")
  set par(
    justify: true,
    leading: 0.68em,
    first-line-indent: 0pt,
    spacing: 0.7em,
  )
  set page(
    paper: "a4",
    margin: (top: 24mm, bottom: 25mm, left: 31mm, right: 27mm),
    header: book-header,
    footer: book-footer,
  )
  set heading(numbering: book-section-numbering)
  set math.equation(numbering: "(1)")

  show heading: it => block(
    width: 100%,
    above: if it.level == 1 { 0pt } else if it.level == 2 { 1.8em } else if it.level == 3 { 1.35em } else { 0.9em },
    below: if it.level <= 2 { 0.55em } else if it.level == 3 { 0.35em } else { 0.25em },
  )[
    #set par(justify: false)
    #if not it.outlined [
      #text(size: 13pt, weight: "semibold")[#it.body]
    ] else if it.level == 1 [
      #align(center)[#text(size: 19pt, weight: "semibold")[#it.body]]
    ] else if it.level == 2 [
      #text(size: 18pt, weight: "semibold", hyphenate: false)[#it.body]
    ] else if it.level == 3 [
      #text(size: 14pt, weight: "semibold")[
        #numbering(it.numbering, ..counter(heading).get()) #h(0.55em) #it.body
      ]
    ] else [
      #text(size: 11.2pt, weight: "semibold")[
        #numbering(it.numbering, ..counter(heading).get()) #h(0.45em) #it.body
      ]
    ]
  ]

  body
}

#let part(number, title) = {
  pagebreak(weak: true)
  page(
    paper: "a4",
    margin: (top: 24mm, bottom: 25mm, left: 31mm, right: 27mm),
    header: none,
    footer: none,
  )[
    #v(38%)
    #heading(level: 1, numbering: none)[Part #number — #title]
  ]
}

#let chapter(part-number, chapter-number, title, body) = {
  pagebreak(weak: true)
  counter(heading).update((part-number, chapter-number))
  heading(level: 2, numbering: none)[Chapter #chapter-number: #title]
  set heading(offset: 2)
  body
}
