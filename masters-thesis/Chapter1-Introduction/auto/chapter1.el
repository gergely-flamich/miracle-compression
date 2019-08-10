(TeX-add-style-hook
 "chapter1"
 (lambda ()
   (LaTeX-add-labels
    "sec:intro_image_compression"
    "sec:intro_distrotion"
    "sec:transform_coding"
    "sec:intro_theoretical_foundations"
    "sec:mdl"
    "eq:min_desc_princ"
    "eq:hypothesis_entropy"
    "eq:hypothesis_cross_entropy"
    "sec:compression_without_quantization"
    "eq:harsha_upper_bound"
    "eq:mdl_elbo"))
 :latex)

