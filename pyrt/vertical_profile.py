import numpy as np


def conrath(altitude, q0, scale_height, nu) -> np.ndarray:
    r"""Make a Conrath profile

    Parameters
    ----------
    altitude
        The altitude [km] at which to construct a Conrath profile.
    q0
        The surface mixing ratio for each of the Conrath profiles.
    scale_height
        The scale height [km] of each of the Conrath profiles.
    nu
        The nu parameter of each of the Conrath profiles.

    Raises
    ------
    TypeError
        Raised if any of the inputs are not a numpy.ndarray.
    ValueError
        Raised if many things...

    Notes
    -----
    The Conrath profile is defined as

    .. math::

       q(z) = q_0 * e^{\nu(1 - e^{z/H})}

    where :math:`q` is a volumetric mixing ratio, :math:`z` is the altitude,
    :math:`\nu` is the Conrath nu parameter, and :math:`H` is the scale
    height.

    """
    altitude_scaling = altitude / scale_height
    return q0 * np.exp(nu * (1 - np.exp(altitude_scaling)))


def uniform(altitude, top, bottom) -> np.ndarray:
    """Make a uniform profile.

    Parameters
    ----------
    altitude
    top
    bottom

    Returns
    -------

    """
    alt_dif = np.diff(altitude, axis=0)
    top_prof = np.clip((top - altitude[1:]) / np.abs(alt_dif), 0, 1)
    bottom_prof = np.clip((altitude[:-1] - bottom) / np.abs(alt_dif), 0, 1)
    return top_prof + bottom_prof - 1

