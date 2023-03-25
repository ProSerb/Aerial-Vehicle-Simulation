"""
Python module filters.py for various filters.
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

import numpy as np
from mobotpy.integration import rk_four


class EKF:
    """EKF filter class.

    Parameters
    ----------
    T : float
        Sampling period [s].
    Q : ndarray
        A priori convariance matrix.
    R : ndarray
        A posteriori convariance matrix.
    """

    def __init__(self, T, Q, R):
        """Constructor method."""
        self.T = T
        self.Q = Q
        self.R = R

    def filter(self, vehicle, sensor, u, x, z, P):

        # Compute the a priori estimate

        # Help the covariance matrix stay symmetric

        # Compute the a posteriori estimate

        # Help the covariance matrix stay symmetric

        # Generate the output
        return x_update, P_update
