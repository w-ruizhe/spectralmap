Quickstart
==========

1. Create a mapping object
--------------------------

.. code-block:: python

   from spectralmap.mapping import Map

   # Rotational mapping (phase curves)
   m_rot = Map(mode="rotational", ydeg=5, inc=90, map_res=30)

   # Eclipse mapping (requires starry bodies)
   # m_ecl = Map(mode="eclipse", pri=pri, sec=sec, map_res=30)

2. Build the design matrix and solve
------------------------------------

.. code-block:: python

   import numpy as np

   theta = np.linspace(0, 360, 200)
   A = m_rot.design_matrix(theta)
   y = np.ones_like(theta)  # replace with your observed flux
   sigma = np.full_like(theta, 1e-4)

   mean, cov, log_ev = m_rot.solve_posterior(
       y=y,
       sigma_y=sigma,
       theta=theta,
       lam_fix=1.0,
   )

3. Where to put your code
-------------------------

- Core algorithms in ``src/spectralmap/``.
- Tutorials in ``docs/source/tutorials/``.
