"""Free Variable."""

from pqueens.distributions.distributions import Distribution


class FreeVariable(Distribution):
    """Free variable class.

    This is not a proper distribution class. It is used for variables
    with no underlying distribution.
    """

    def __init__(self, dimension):
        """Initialize FreeVariable object.

        Args:
            dimension (int): Dimensionality of the variable
        """
        super().__init__(mean=None, covariance=None, dimension=dimension)

    @classmethod
    def from_config_create_distribution(cls, distribution_options):
        """Create FreeVariable object from parameter dictionary.

        Args:
            distribution_options (dict): Dictionary with distribution description

        Returns:
            distribution: FreeVariable object
        """
        dimension = distribution_options.get('lower_bound', 1)
        return cls(dimension=dimension)

    def cdf(self, _):
        """Cumulative distribution function."""
        raise ValueError('cdf method is not supported for FreeVariable.')

    def draw(self, _):
        """Draw samples."""
        raise ValueError('draw method is not supported for FreeVariable.')

    def logpdf(self, _):
        """Log of the probability density function."""
        raise ValueError('logpdf method is not supported for FreeVariable.')

    def grad_logpdf(self, _):
        """Gradient of the log pdf with respect to *x*."""
        raise ValueError('grad_logpdf method is not supported for FreeVariable.')

    def pdf(self, _):
        """Probability density function."""
        raise ValueError('pdf method is not supported for FreeVariable.')

    def ppf(self, _):
        """Percent point function (inverse of cdf — quantiles)."""
        raise ValueError('ppf method is not supported for FreeVariable.')