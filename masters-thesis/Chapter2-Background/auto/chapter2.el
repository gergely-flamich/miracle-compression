(TeX-add-style-hook
 "chapter2"
 (lambda ()
   (LaTeX-add-labels
    "chapter:background"
    "sec:intro_image_compression"
    "sec:intro_distrotion"
    "sec:transform_coding"
    "sec:intro_theoretical_foundations"
    "sec:mdl"
    "eq:min_desc_princ"
    "eq:hypothesis_entropy"
    "eq:hypothesis_cross_entropy"
    "sec:compression_without_quantization"
    "thm:bits-back_efficiency"
    "eq:harsha_upper_bound"
    "eq:mdl_elbo"
    "sec:derive_weighted_elbo"
    "eq:framework_hard_train_target"
    "eq:framework_train_target"))
 :latex)

