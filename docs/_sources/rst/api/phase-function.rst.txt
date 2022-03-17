Phase Function
==============

.. currentmodule:: pyrt

.. note::
   I'm using the phrase "phase function" to denote an array that is dependent
   only on scattering angle. You can reasonably argue a phase function is
   dependent on particle size and wavelength. I simply refer to this case as
   an "array of phase functions".

Generic phase function
----------------------
.. autosummary::
   :toctree: generated/

   resample
   normalize
   decompose
   fit_asymmetry_parameter

Henyey-Greenstein
-----------------
.. autosummary::
   :toctree: generated/

   construct_hg
   decompose_hg

Legendre coefficients
---------------------
.. autosummary::
   :toctree: generated/

   set_negative_coefficients_to_0
   reconstruct