(TeX-add-style-hook
 "chapter3"
 (lambda ()
   (LaTeX-add-labels
    "chapter:related_works"
    "sec:lit_comparison"
    "sec:related_works_datasets"
    "eq:gdn_def"
    "eq:igdn_def"
    "fig:comp_auto_arch"
    "fig:rippel_arch"
    "fig:balle_ladder_arch"
    "sec:comp_quant"
    "fig:quantization_models"
    "eq:quantization_step"
    "fig:rippel_pipeline"
    "eq:balle_var_train_objective"))
 :latex)

