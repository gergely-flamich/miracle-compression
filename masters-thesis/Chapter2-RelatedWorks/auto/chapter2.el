(TeX-add-style-hook
 "chapter2"
 (lambda ()
   (LaTeX-add-labels
    "chapter:related_works"
    "sec:lit_comparison"
    "eq:gdn_def"
    "eq:igdn_def"
    "sec:comp_quant"
    "eq:balle_var_train_objective"))
 :latex)

