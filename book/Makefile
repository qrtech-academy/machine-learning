# Build the book: two LuaLaTeX passes, the second for the contents and the references.
#
#   make                   build machine-learning.pdf, dated today
#   make VERSION=book-v2   the same, with the release's version on the title page
#   make png               redraw the lecture's copies of the network figures from figures/
#   make clean             remove everything the build writes
#
# The interfaces the exercises in Chapters 8 and 10 implement are typeset straight from their
# files under lectures/, so a change to one of those is a change to the book on the next build.

ROOT    := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
BOOK    := $(ROOT)/book
BUILD   := $(BOOK)/build
VERSION ?=
LATEX   := lualatex -interaction=nonstopmode -halt-on-error -file-line-error \
           -output-directory=$(BUILD)
# The book, with its version defined before anything is read; -jobname keeps the output book.pdf.
BOOKTEX := $(LATEX) -jobname=book '\def\bookversion{$(VERSION)}\input{book.tex}'

SOURCES := book.tex mlbook.sty mlbook.lua \
           $(wildcard front/*.tex chapters/*/*.tex back/*/*.tex figures/*.tex)
CODE    := $(shell cd $(ROOT) && git ls-files 'lectures/*.cpp' 'lectures/*.hpp' \
             'lectures/*Makefile' 'lectures/*makefile' | sed 's|^|$(ROOT)/|')

# The lecture's PNG of each network figure: L03's third example uses the same network as its
# first, so network3.png is the drawing of network1.
IMAGES  := $(ROOT)/lectures/L03/appendix/images
PNGS    := network1:network1 network2:network2 network1:network3

.PHONY: all png clean FORCE

all: machine-learning.pdf

# The version the last build was made for. The file is rewritten only when VERSION changes, so a
# new version, or none after one, rebuilds the title page and nothing else does.
$(BUILD)/version: FORCE
	@mkdir -p $(BUILD)
	@echo '$(VERSION)' | cmp -s - $@ || echo '$(VERSION)' > $@

machine-learning.pdf: $(SOURCES) $(CODE) $(BUILD)/version
	@mkdir -p $(BUILD)
	cd $(BOOK) && $(BOOKTEX) >/dev/null || (tail -40 $(BUILD)/book.log; exit 1)
	cd $(BOOK) && $(BOOKTEX) >/dev/null || (tail -40 $(BUILD)/book.log; exit 1)
	@grep -E "^(Overfull|Underfull)|Warning" $(BUILD)/book.log | sort | uniq -c | sort -rn | head -20 || true
	@! grep -q "There were undefined references" $(BUILD)/book.log || \
	    (echo "error: undefined references; see $(BUILD)/book.log"; exit 1)
	cp $(BUILD)/book.pdf $@

# L03 shows the same drawings as the book, as PNGs; this is the one command that makes them.
png:
	@mkdir -p $(BUILD)
	@for pair in $(PNGS); do \
	    figure=$${pair%%:*}; image=$${pair##*:}; \
	    echo "figures/$$figure.tex -> $(IMAGES)/$$image.png"; \
	    (cd $(BOOK) && $(LATEX) -jobname=png-$$figure \
	        "\def\pngfigure{$$figure}\input{figures/png.tex}" >/dev/null) || \
	        (tail -40 $(BUILD)/png-$$figure.log; exit 1); \
	    pdftoppm -r 200 -png -singlefile $(BUILD)/png-$$figure.pdf $(IMAGES)/$$image || exit 1; \
	done

clean:
	rm -rf $(BUILD) machine-learning.pdf
