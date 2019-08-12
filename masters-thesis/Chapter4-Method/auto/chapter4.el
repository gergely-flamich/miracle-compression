(TeX-add-style-hook
 "chapter4"
 (lambda ()
   (LaTeX-add-labels
    "chapter:method"
    "sec:dataset_preproc"
    "sec:architectures"
    "sec:method_vaes"
    "fig:vae_rand_posterior"
    "fig:zhao_loss_comparison"
    "eq:regular_vae_elbo"
    "eq:laplace_likelihood"
    "sec:prob_ladder_networks"
    "fig:pln_architecture"
    "fig:ladder_rand_posterior"
    "sec:learn_gamma"
    "fig:gamma_rand_posterior"
    "sec:coded_sampling"
    "alg:multivariate_rej_samp"
    "alg:greedy_sampler"
    "alg:adaptive_importance_sampler"))
 :latex)

