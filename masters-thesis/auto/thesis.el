(TeX-add-style-hook
 "thesis"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("PhDThesisPSnPDF" "a4paper" "12pt" "times" "authoryear" "print" "index")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "Preamble/preamble"
    "thesis-info"
    "Dedication/dedication"
    "Declaration/declaration"
    "Acknowledgement/acknowledgement"
    "Abstract/abstract"
    "Chapter1-Introduction/chapter1"
    "Chapter2-Background/chapter2"
    "Chapter3-RelatedWorks/chapter3"
    "Chapter4-Method/chapter4"
    "Chapter5-Experiments/chapter5"
    "Chapter6-Conclusion/chapter6"
    "Appendix1/appendix1"
    "Appendix2/appendix2"
    "PhDThesisPSnPDF"
    "PhDThesisPSnPDF12")
   (LaTeX-add-bibliographies
    "References/references"))
 :latex)

