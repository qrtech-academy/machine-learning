# The Book

The course typeset as a book with LuaLaTeX: ten chapters, one per lecture, in the course's two
parts, then the worked solutions to every exercise answered on paper and the two self-assessment
papers as appendices. The solutions to the code exercises stay in the repository.

---

## Building it

```bash
sudo apt -y install make texlive-luatex texlive-latex-extra fonts-texgyre fonts-texgyre-math \
                    fonts-dejavu-core poppler-utils
make -C book                   # Writes book/machine-learning.pdf, dated today.
make -C book VERSION=book-v2   # The same, with the version on the title page.
make -C book png               # Redraws L03's network PNGs from the figures.
make -C book clean             # Removes book/build/ and the PDF.
```

The build runs LuaLaTeX twice, so the contents and the cross-references settle, then prints any
overfull or underfull lines and LaTeX warnings it found, and fails if a reference is left undefined.
A clean build prints nothing after the two `lualatex` lines except a handful of mildly underfull
ones.

---

## Releasing a new edition

The PDF is committed, as `book/machine-learning.pdf`, so the repository always holds a readable
copy; rebuild it with `make -C book` and commit it along with any change to the book. Each edition
is also published as a GitHub release, with its version on the title page. Push a tag named
`book-v*`:

```bash
git tag book-v1
git push origin book-v1
```

The [Book workflow](../.github/workflows/book.yml) then builds the PDF with the tag on its title
page and attaches it to a release of the same name. The README's download link always points at
the newest release, so it never needs updating.

---

## What is where

```text
book.tex                The book: front matter, two parts of five chapters, appendices, in order.
mlbook.sty              Every visual decision: page, type, colours, code blocks, exercises.
mlbook.lua              How \code{...} typesets inline C++ (#, \n and line breaks).
front/                  Title pages and preface.
chapters/NN/            Chapter NN: chapter.tex (the opener), one file per appendix of lecture
                        LNN, summary.tex (the review) and exercises.tex.
back/solutions/         Appendix A: the worked solutions, one file per chapter that has any.
back/exam/              Appendices B to F: the papers and their model answers.
figures/                Figures drawn in TikZ, and png.tex, which renders one as the lecture's PNG.
```

Every `.tex` file typeset from course material starts with a comment naming its source, for
example:

```tex
% Sections 1.1 to 1.4, from lectures/L01/appendix/a_theory.md: one section per numbered section of
% the appendix.
```

---

## Updating the content

The course material is the source of truth, and the book follows it. **Two kinds of content behave
differently:**

* **The interfaces in Chapters 8 and 10 update themselves.** The book does not contain their code;
  it includes the headers under `lectures/L08/cnn_work/include` directly
  (`\cppfile{lectures/...}`), so a change to one is in the book on the next build, with nothing to
  edit here.
* **Prose and the code snippets in the text do not.** A chapter's text is a typeset copy of its
  lecture's markdown. When you change a lecture appendix, make the same change in the `.tex` file
  whose header names it. The same holds for `exam/*.md` and `back/exam/`, and for the solutions
  README of L01 and `back/solutions/chapter01.tex`.

A few conventions, so an edit reads like the rest of the book:

* Code blocks: `cppcode` (C++), `ccode` (C), `pycode` (the Python-style pseudocode), `shell`,
  `makecode` (Makefiles; recipe lines keep their tab), and `console` (program output, matrices,
  directory trees, plain text). Box drawing, arrows and Greek letters work inside all of them; a
  character the design has not registered yet is added to the list in `mlbook.sty`, beside the
  comment that explains why listings needs it.
* Inline code: `\code{...}`, written exactly as in the source. Inside it, write `\%` for `%`,
  `\{` or `\}` for an unbalanced brace, `\\` for a backslash, and `\#` for `#` in a heading or
  caption.
* Maths: `\LR` for the learning rate, `\yref` and `\yp` for the reference and predicted values,
  `\MAE` for the mean absolute error, and `\cdot` where the lectures write `*`.
* References: `\secref{c2:sec:precision}` is a section, `Chapter~\ref{c3:ch:nn}` a chapter (a
  lecture mentioned as a unit of teaching, "in L03", becomes its chapter), and
  `Exercise~\ref{c4:ex:3}` is Exercise 4.3. The book numbers exercises within each chapter, so
  always refer to them by label; `cN:ex:K` is the label of Exercise N.K.
* Exercises: `\exerciseset{Title}` for an exercise set, then `\codeexercise{Title}`,
  `\calcexercise{Title}` or `\reflexercise{Title}` for one exercise, and `\task{...}` for a titled
  part.
* Worked solutions: an exercise answered on paper ends with `\seesolution{cN:ex:K}`, and its
  solution in `back/solutions/chapterNN.tex` starts with `\solution{cN:ex:K}{Title}`; the two point
  at each other by page. Code exercises have no solution in the book.
* A new lecture appendix is a new file in `chapters/NN/`, `\input` from that chapter's
  `chapter.tex`.

The solutions to the code exercises are not in the book. Each chapter's exercises say where they are
in the repository, so moving or renaming a solution directory means updating that chapter's
`exercises.tex`.

The two network figures in Chapter 3 are also the course's PNGs. Edit `figures/network1.tex` or
`figures/network2.tex`, then run `make -C book png` to redraw `lectures/L03/appendix/images/`, so
the lecture and the book stay the same drawing. L03's third example uses the same network as its
first, so `network3.png` is drawn from `network1.tex`.

---

## License

The book, its text and figures and the PDF built from them, is licensed under
[CC BY-NC-SA 4.0](../LICENSE-CONTENT), like the course material it is typeset from. This directory's
build files (`mlbook.sty`, `mlbook.lua`, `Makefile`) are released under the repository's
[MIT License](../LICENSE), like all the code in the course, and so may the code examples printed in
the book be used.
