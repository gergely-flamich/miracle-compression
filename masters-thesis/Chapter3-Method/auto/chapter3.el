(TeX-add-style-hook
 "chapter3"
 (lambda ()
   (LaTeX-add-labels
    "chapter:method"
    "sec:our_method"
    "sec:dataset_preproc"
    "sec:architectures"
    "fig:vae_rand_posterior"
    "eq:miracle_hard_train_target"
    "eq:miracle_train_target"
    "eq:miracle_ub"
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
    "alg:adaptive_importance_sampler"
    "sec:entropy_coding"
    "sec:rej_samp_artihmetic_coding"))
 :latex)

