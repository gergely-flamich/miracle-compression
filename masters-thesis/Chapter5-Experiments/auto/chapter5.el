(TeX-add-style-hook
 "chapter5"
 (lambda ()
   (LaTeX-add-labels
    "chapter:experiments"
    "sec:experimental_results"
    "fig:pln_reconstruction"
    "fig:gamma_reconstruction"
    "fig:kodim05_comp"
    "fig:kodim05_side_info"
    "fig:kodim05_coding_time"
    "tab:kodim05_coding_time"))
 :latex)

