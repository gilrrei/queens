"""Gaussian distributions."""
import numpy as np

from queens.distributions.normal import NormalDistribution

# pylint: disable=invalid-name


# 1d standard Gaussian
standard_normal = NormalDistribution(0.0, 1)

# 2d Gaussian
dim = 2

mean = [0.0, 0.0]
cov = [[1.0, 0.5], [0.5, 1.0]]

A = np.eye(dim, dim)
b = np.zeros(dim)

gaussian_2d = NormalDistribution(mean, cov)

# 4d Gaussian
cov = [
    [2.691259143915389, 1.465825570809310, 0.347698874175537, 0.140030644426489],
    [1.465825570809310, 4.161662217930926, 0.423882544003853, 1.357386322235196],
    [0.347698874175537, 0.423882544003853, 2.928845742295657, 0.484200164430076],
    [0.140030644426489, 1.357386322235196, 0.484200164430076, 3.350315448057768],
]

mean = [0.806500709319150, 2.750827521892630, -3.388270291505472, 1.293259980552181]

gaussian_4d = NormalDistribution(mean, cov)


def gaussian_1d_logpdf(x):
    """1D Gaussian likelihood model.

    Used as a basic test function for MCMC methods.

    Returns:
        float: The logpdf evaluated at *x*
    """
    y = np.atleast_2d(standard_normal.logpdf(x))
    return y


def gaussian_2d_logpdf(samples):
    """2D Gaussian logpdf.

    Args:
        samples (np.ndarray): Samples to be evaluated

    Returns:
        np.ndarray: logpdf
    """
    model_data = np.dot(A, samples.T).T + b
    y = gaussian_2d.logpdf(model_data)
    return y


def gaussian_4d_logpdf(samples):
    """4D Gaussian logpdf.

    Args:
        samples (np.ndarray): Samples to be evaluated

    Returns:
        np.ndarray: logpdf
    """
    y = gaussian_4d.logpdf(samples)
    return y