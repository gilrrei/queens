"""No-U-Turn algorithm.

"The No-U-Turn sampler is a gradient based MCMC algortihm. It builds on
the Hamiltonian Monte Carlo sampler to sample from (high dimensional)
arbitrary probability distributions.
"""

import logging

import pymc as pm

from pqueens.iterators.pymc_iterator import PyMCIterator
from pqueens.utils.pymc import PymcDistributionRapper

_logger = logging.getLogger(__name__)


class NUTSIterator(PyMCIterator):
    """Iterator based on HMC algorithm.

    References:
        [1]: Hoffman et al. The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian
        Monte Carlo. 2011.

    The No-U-Turn sampler is a state of the art MCMC sampler. It is based on the Hamiltonian Monte
    Carlo sampler but eliminates the need for an specificed number of integration step by checking
    if the trajectory turns around. The algorithm is based on a building up a tree and selecting a
    random note as proposal.

    Attributes:
        max_treedepth (int): Maximum depth for the tree-search
        target_accept (float): Target accpetance rate which should be conistent after burn-in
        scaling (np.array): The inverse mass, or precision matrix
        is_cov (boolean): Setting if the scaling is a mass or covariance matrix
    Returns:
        nuts_iterator (obj): Instance of NUTS Iterator
    """

    def __init__(
        self,
        global_settings,
        model,
        num_burn_in,
        num_chains,
        num_samples,
        init_strategy,
        discard_tuned_samples,
        result_description,
        seed,
        use_queens_prior,
        progressbar,
        max_treedepth,
        target_accept,
        scaling,
        is_cov,
    ):
        """Initialize NUTS iterator.

        Args:
            global_settings (dict): Global settings of the QUEENS simulations
            model (obj): Underlying simulation model on which the inverse analysis is conducted
            num_burn_in (int): Number of burn-in steps
            num_chains (int): Number of chains to sample
            num_samples (int): Number of samples to generate per chain, excluding burn-in period
            init_strategy (str): Strategy to tune mass damping matrix
            discard_tuned_samples (boolean): Setting to discard the samples of the burin-in period
            result_description (dict): Settings for storing and visualizing the results
            seed (int): Seed for rng
            use_queens_prior (boolean): Setting for using the PyMC priors or the QUEENS prior
            functions
            progressbar (boolean): Setting for printing progress bar while sampling
            max_treedepth (int): Maximum depth for the tree-search
            target_accept (float): Target accpetance rate which should be conistent after burn-in
            scaling (np.array): The inverse mass, or precision matrix
            is_cov (boolean): Setting if the scaling is a mass or covariance matrix
        Returns:
            Initialise pymc iterator
        """
        super().__init__(
            global_settings,
            model,
            num_burn_in,
            num_chains,
            num_samples,
            init_strategy,
            discard_tuned_samples,
            result_description,
            seed,
            use_queens_prior,
            progressbar,
        )

        self.max_treedepth = max_treedepth
        self.target_accept = target_accept
        self.scaling = scaling
        self.is_cov = is_cov

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create NUTS iterator from problem description.

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator:NUTSIterator object
        """
        _logger.info(
            "NUTS Iterator for experiment: {0}".format(
                config.get('global_settings').get('experiment_name')
            )
        )

        method_options = config[iterator_name]['method_options']

        (
            global_settings,
            model,
            num_burn_in,
            num_chains,
            num_samples,
            init_strategy,
            discard_tuned_samples,
            result_description,
            seed,
            use_queens_prior,
            progressbar,
        ) = super().get_base_attributes_from_config(config, iterator_name)

        max_treedepth = method_options.get('max_treedepth', 10)
        target_accept = method_options.get('target_accept', 0.8)
        scaling = method_options.get('scaling', None)
        is_cov = method_options.get('is_cov', False)

        return cls(
            global_settings=global_settings,
            model=model,
            num_burn_in=num_burn_in,
            num_chains=num_chains,
            num_samples=num_samples,
            init_strategy=init_strategy,
            discard_tuned_samples=discard_tuned_samples,
            result_description=result_description,
            seed=seed,
            use_queens_prior=use_queens_prior,
            progressbar=progressbar,
            max_treedepth=max_treedepth,
            target_accept=target_accept,
            scaling=scaling,
            is_cov=is_cov,
        )

    def init_mcmc_method(self):
        """Init the PyMC MCMC Model.

        Args:

        Returns:
            step (obj): The MCMC Method within the PyMC Model
        """
        step = pm.NUTS(
            target_accept=self.target_accept,
            max_treedepth=self.max_treedepth,
            scaling=self.scaling,
            is_cov=self.is_cov,
        )
        return step

    def init_distribution_wrapper(self):
        """Init the PyMC wrapper for the QUEENS distributions."""
        self.loglike = PymcDistributionRapper(
            self.eval_log_likelihood, self.eval_log_likelihood_grad
        )
        if self.use_queens_prior:
            self.logprior = PymcDistributionRapper(self.eval_log_prior, self.eval_log_prior_grad)
        _logger.info("Initialize NTUS by PyMC run.")
