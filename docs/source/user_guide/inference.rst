Inference
=========

Problem setup
-------------

Optimization (MAP)
------------------

Uncertainty estimation
----------------------

Sampling (MCMC / VI)
--------------------

Posterior predictive checks
---------------------------

Regularizer toggles
-------------------

SpectralMap exposes two switches in ``Map.solve_posterior``:

- ``use_alpha``: enable/disable coefficient-prior regularization.
- ``use_lambda``: enable/disable intensity-space regularization.

Example:

.. code-block:: python

   mean, cov, log_ev = map.solve_posterior(
	   y=y,
	   sigma_y=sigma,
	   theta=theta,
	   use_alpha=True,
	   use_lambda=False,
	   lam_fix=1.0,
   )

Evidence accounting
~~~~~~~~~~~~~~~~~~~

When a regularizer is disabled, its contribution is removed from evidence terms:

- ``use_alpha=False``: no alpha prior/evidence correction term.
- ``use_lambda=False``: no lambda shrinkage/evidence correction term.

This keeps ``log_ev`` and ``log_ev_marginalized`` consistent with the active
model only, so comparisons across runs are interpretable by construction.

