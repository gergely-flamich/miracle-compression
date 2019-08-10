(TeX-add-style-hook
 "chapter1"
 (lambda ()
   (LaTeX-add-labels
    "sec:theoretical_foundations"
    "sec:mdl"
    "eq:min_desc_princ"
    "eq:hypothesis_entropy"
    "eq:hypothesis_cross_entropy"
    "eq:mdl_elbo"))
 :latex)

