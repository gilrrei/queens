import numpy as np

from pqueens.utils import mcmc_utils

cov = [
    [2.691259143915389, 1.465825570809310, 0.347698874175537, 0.140030644426489],
    [1.465825570809310, 4.161662217930926, 0.423882544003853, 1.357386322235196],
    [0.347698874175537, 0.423882544003853, 2.928845742295657, 0.484200164430076],
    [0.140030644426489, 1.357386322235196, 0.484200164430076, 3.350315448057768],
]

mean = [0.806500709319150, 2.750827521892630, -3.388270291505472, 1.293259980552181]

distr_dict = {'distribution': 'normal', 'distribution_parameter': [mean, cov]}

gaussian = mcmc_utils.create_proposal_distribution(distr_dict)


def gaussian_logpdf(x1, x2, x3, x4):
    """ 4D Gaussian likelihood model

    Used as a basic test function for SMC.

    The log likelihood is defined as (see [1]):

    :math:`f({x}) = -\\frac{1}{2}(x-\\mu)^T\\Sigma^{-1}(x-\\mu) - \\log(\\sqrt((2 \\pi)^k |\\Sigma|)`

    Args:
        x1 (float):
        x2 (float):
        x3 (float):
        x4 (float):

    Returns:
        numpy.array : The logpdf evaluated at x


    References:

        [1] https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    """

    x = np.array([x1, x2, x3, x4])

    return gaussian.logpdf(x)


def main(job_id, params):
    """ Interface to 1D Guassian model

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters
    Returns:
        float: Value of Gaussian at parameters specified in input dict
    """
    return gaussian_logpdf(params['x1'], params['x2'], params['x3'], params['x4'])