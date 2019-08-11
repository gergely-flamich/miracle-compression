(TeX-add-style-hook
 "thesis"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("PhDThesisPSnPDF" "a4paper" "12pt" "times" "authoryear" "print" "index")))
   (TeX-run-style-hooks
    "latex2e"
    "Preamble/preamble"
    "thesis-info"
    "Dedication/dedication"
    "Declaration/declaration"
    "Acknowledgement/acknowledgement"
    "Abstract/abstract"
    "Chapter1-Introduction/chapter1"
    "Chapter2-RelatedWorks/chapter2"
    "Chapter3-Method/chapter3"
    "Chapter4-Experiments/chapter4"
    "Chapter5-Conclusion/chapter5"
    "Appendix1/appendix1"
    "Appendix2/appendix2"
    "PhDThesisPSnPDF"
    "PhDThesisPSnPDF12")
   (LaTeX-add-bibliographies
    "References/references"))
 :latex)

