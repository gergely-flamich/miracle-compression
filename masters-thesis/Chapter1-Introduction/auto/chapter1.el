(TeX-add-style-hook
 "chapter1"
 (lambda ()
   (LaTeX-add-labels
    "sec:theoretical_foundations"
    "sec:mdl"
    "eq:min_desc_princ"
    "eq:elbo_target"
    "sec:miracle_theory"
    "eq:miracle_hard_train_target"
    "eq:miracle_train_target"
    "eq:miracle_ub"))
 :latex)

