"""Iterators.

The iterator package contains the implementation of several UQ and
optimization methods, each of which is implemented in their own iterator
class. The iterator is therefor one of the central building blocks, as
the iterators orchestrate the evaluations on one or multiple models.
QUEENS also permits nesting of iterators to enable hierarchical methods
or surrogate based UQ approaches.
"""

from queens.iterators.baci_lm_iterator import BaciLMIterator
from queens.iterators.black_box_variational_bayes import BBVIIterator
from queens.iterators.bmfia_iterator import BMFIAIterator
from queens.iterators.bmfmc_iterator import BMFMCIterator
from queens.iterators.classification import ClassificationIterator
from queens.iterators.data_iterator import DataIterator
from queens.iterators.elementary_effects_iterator import ElementaryEffectsIterator
from queens.iterators.grid_iterator import GridIterator
from queens.iterators.hmc_iterator import HMCIterator
from queens.iterators.lhs_iterator import LHSIterator
from queens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from queens.iterators.metropolis_hastings_pymc_iterator import MetropolisHastingsPyMCIterator
from queens.iterators.monte_carlo_iterator import MonteCarloIterator
from queens.iterators.nuts_iterator import NUTSIterator
from queens.iterators.optimization_iterator import OptimizationIterator
from queens.iterators.points_iterator import PointsIterator
from queens.iterators.polynomial_chaos_iterator import PolynomialChaosIterator
from queens.iterators.reparameteriztion_based_variational_inference import RPVIIterator
from queens.iterators.sequential_monte_carlo_chopin import SequentialMonteCarloChopinIterator
from queens.iterators.sequential_monte_carlo_iterator import SequentialMonteCarloIterator
from queens.iterators.sobol_index_gp_uncertainty_iterator import SobolIndexGPUncertaintyIterator
from queens.iterators.sobol_index_iterator import SobolIndexIterator
from queens.iterators.sobol_sequence_iterator import SobolSequenceIterator

VALID_TYPES = {
    'hmc': HMCIterator,
    'lhs': LHSIterator,
    'metropolis_hastings': MetropolisHastingsIterator,
    'metropolis_hastings_pymc': MetropolisHastingsPyMCIterator,
    'monte_carlo': MonteCarloIterator,
    'nuts': NUTSIterator,
    'optimization': OptimizationIterator,
    'read_data_from_file': DataIterator,
    'elementary_effects': ElementaryEffectsIterator,
    'polynomial_chaos': PolynomialChaosIterator,
    'sobol_indices': SobolIndexIterator,
    'sobol_indices_gp_uncertainty': SobolIndexGPUncertaintyIterator,
    'smc': SequentialMonteCarloIterator,
    'smc_chopin': SequentialMonteCarloChopinIterator,
    'sobol_sequence': SobolSequenceIterator,
    'points': PointsIterator,
    'bmfmc': BMFMCIterator,
    'grid': GridIterator,
    'baci_lm': BaciLMIterator,
    'bbvi': BBVIIterator,
    'bmfia': BMFIAIterator,
    'rpvi': RPVIIterator,
    'classification': ClassificationIterator,
}