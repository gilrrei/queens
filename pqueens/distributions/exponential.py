"""Exponential distribution."""
import numpy as np

from pqueens.distributions.distributions import Distribution


class ExponentialDistribution(Distribution):
    r"""Exponential distribution class.

    For a multivariate distribution the components are assumed to be independent.

    Attributes:
        rate (np.ndarray): Rate parameter(s) of the distribution.
        scale (np.ndarray): Scale parameters(s) of the distribution
                            (:math:`scale = \frac{1}{rate}`) .
        pdf_const (float): Constant for the evaluation of the pdf.
        logpdf_const (float): Constant for the evaluation of the log pdf.
    """

    def __init__(self, rate, scale, pdf_const, logpdf_const, mean, covariance, dimension):
        """Initialize exponential distribution.

        Args:
            rate (np.ndarray): rate parameter(s) of the distribution
            scale (np.ndarray): scale parameters(s) of the distribution (scale = 1 / rate)
            pdf_const (float): Constant for the evaluation of the pdf
            logpdf_const (float): Constant for the evaluation of the log pdf
            mean (np.ndarray): Mean of the distribution
            covariance (np.ndarray): Covariance of the distribution
            dimension (int): Dimensionality of the distribution
        """
        super().__init__(mean=mean, covariance=covariance, dimension=dimension)
        self.rate = rate
        self.scale = scale
        self.pdf_const = pdf_const
        self.logpdf_const = logpdf_const

    @classmethod
    def from_config_create_distribution(cls, distribution_options):
        """Create exponential distribution object from parameter dictionary.

        Args:
            distribution_options (dict): Dictionary with distribution description

        Returns:
            distribution: ExponentialDistribution object
        """
        rate = np.array(distribution_options['rate']).reshape(-1)
        scale = 1 / rate
        super().check_positivity({'rate': rate})

        mean = scale
        covariance = np.diag(scale**2)
        dimension = mean.size

        pdf_const = np.prod(rate)
        logpdf_const = np.sum(np.log(rate))

        return cls(
            rate=rate,
            scale=scale,
            pdf_const=pdf_const,
            logpdf_const=logpdf_const,
            mean=mean,
            covariance=covariance,
            dimension=dimension,
        )

    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated

        Returns:
            cdf (np.ndarray): cdf at evaluated positions
        """
        x = x.reshape(-1, self.dimension)
        condition = (x >= 0).all(axis=1)
        cdf = np.where(condition, np.prod(1 - np.exp(-self.rate * x), axis=1), 0)
        return cdf

    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws

        Returns:
            samples (np.ndarray): Drawn samples from the distribution
        """
        samples = np.random.exponential(scale=self.scale, size=(num_draws, self.dimension))
        return samples

    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated

        Returns:
            logpdf (np.ndarray): log pdf at evaluated positions
        """
        x = x.reshape(-1, self.dimension)
        condition = (x >= 0).all(axis=1)
        logpdf = self.logpdf_const + np.where(condition, np.sum(-self.rate * x, axis=1), -np.inf)
        return logpdf

    def grad_logpdf(self, x):
        """Gradient of the log pdf with respect to *x*.

        Args:
            x (np.ndarray): Positions at which the gradient of log pdf is evaluated

        Returns:
            grad_logpdf (np.ndarray): Gradient of the log pdf evaluated at positions
        """
        x = x.reshape(-1, self.dimension)
        condition = (x >= 0).all(axis=1).reshape(-1, 1)
        grad_logpdf = np.where(condition, -self.rate, np.nan)
        return grad_logpdf

    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated

        Returns:
            pdf (np.ndarray): Pdf at evaluated positions
        """
        pdf = np.exp(self.logpdf(x))
        return pdf

    def ppf(self, q):
        """Percent point function (inverse of cdf — quantiles).

        Args:
            q (np.ndarray): Quantiles at which the ppf is evaluated

        Returns:
            ppf (np.ndarray): Positions which correspond to given quantiles
        """
        self.check_1d()
        ppf = -self.scale * np.log(1 - q)
        return ppf