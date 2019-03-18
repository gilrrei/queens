import scipy.stats

def gaussian_logpdf(x):
    """ 1D Gaussian likelihood model

    Used as a basic test function for MCMC methods.

    The log likelihood is defined as (see [1]):

    :math:`f({x}) = \\frac{-(x-\\mu)^2}{2\\sigma^2} - \\log(\\sqrt(2 \\pi \\sigma^2)`

    Args:
        x (float):

    Returns:
        float : The logpdf evaluated at x


    References:

        [1] https://en.wikipedia.org/wiki/Normal_distribution
    """
    mean = 0.0
    covariance = 1.0

    y = scipy.stats.norm.logpdf(x, loc = mean, scale = covariance)

    return y


def main(job_id, params):
    """ Interface to 1D Guassian model

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters
    Returns:
        float: Value of Gaussian at parameters specified in input dict
    """
    return gaussian_logpdf(params['x'])
