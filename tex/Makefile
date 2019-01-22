.PHONY : clean, default, publish

REPO_NAME   = $(echo $TRAVIS_REPO_SLUG | cut -f2 -d/) 
TEX_FILES   = eneloop
TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out ent
LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
SUFF        = pdf
RM_TMP      = $(foreach d, ${TEX_FILES}, rm -rf $(foreach suff, ${TMP_SUFFS}, ${d}.${suff})) eneloopNotes.bib
CHECK_RERUN = grep Rerun eneloop.log
FIGURES     = $(patsubst %.py,%.pdf,$(wildcard figures/*.py))
TECTONIC    = $(shell command -v tectonic >/dev/null && echo true || echo false )

default: eneloop.pdf

eneloop.pdf: eneloop.tex eneloop.bib $(FIGURES)
	# Generate links to current git commit
	#python gen_links.py
	# Compile the paper with either tectonic or pdflatex
	if [ "${TECTONIC}" = "true" ]; then\
		tectonic eneloop.tex --print --keep-logs;\
	else\
		${LATEX} eneloop.tex;\
		( ${CHECK_RERUN} && ${LATEX} eneloop.tex ) || echo "Done.";\
		( ${CHECK_RERUN} && ${LATEX} eneloop.tex ) || echo "Done.";\
		( ${CHECK_RERUN} && ${LATEX} eneloop.tex ) || echo "Done.";\
	fi

clean:
	# Remove temporaries and output
	$(RM_TMP)
	rm -rf $(FIGURES)

figures/%.pdf: figures/%.py
	# Generate all figures
	cd $(<D);python $(<F)
