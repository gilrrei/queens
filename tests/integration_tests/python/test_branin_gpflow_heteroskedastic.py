"""Integration test for the GPflow based heteroskedastic model."""

import numpy as np
import pytest

from queens.distributions.uniform import UniformDistribution
from queens.drivers.function_driver import FunctionDriver
from queens.interfaces.job_interface import JobInterface
from queens.iterators.monte_carlo_iterator import MonteCarloIterator
from queens.main import run_iterator
from queens.models import HeteroskedasticGPModel
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.io_utils import load_result


@pytest.mark.max_time_for_test(30)
def test_branin_gpflow_heteroskedastic(expected_mean, expected_var, global_settings):
    """Test case for GPflow based heteroskedastic model."""
    # Parameters
    x1 = UniformDistribution(lower_bound=-5, upper_bound=10)
    x2 = UniformDistribution(lower_bound=0, upper_bound=15)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    driver = FunctionDriver(function="branin78_hifi")
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)
    interface = JobInterface(parameters=parameters, scheduler=scheduler, driver=driver)
    model = SimulationModel(interface=interface)
    training_iterator = MonteCarloIterator(
        seed=42,
        num_samples=100,
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )
    model = HeteroskedasticGPModel(
        eval_fit=None,
        error_measures=[
            "sum_squared",
            "mean_squared",
            "root_mean_squared",
            "sum_abs",
            "mean_abs",
            "abs_max",
        ],
        num_posterior_samples=None,
        num_inducing_points=30,
        num_epochs=100,
        adams_training_rate=0.1,
        random_seed=1,
        num_samples_stats=1000,
        training_iterator=training_iterator,
    )

    iterator = MonteCarloIterator(
        seed=44,
        num_samples=10,
        result_description={
            "write_results": True,
            "plot_results": False,
            "bayesian": False,
            "num_support_points": 10,
            "estimate_all": False,
        },
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["result"], expected_mean, decimal=2
    )
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["variance"], expected_var, decimal=2
    )


@pytest.fixture(name="expected_mean")
def fixture_expected_mean():
    """Expected mean values."""
    mean = np.array(
        [
            [
                5.12898,
                4.07712,
                10.22693,
                2.55123,
                4.56184,
                2.45215,
                2.56100,
                3.32164,
                7.84209,
                6.96919,
            ]
        ]
    ).T
    return mean


@pytest.fixture(name="expected_var")
def fixture_expected_var():
    """Expected variance values."""
    var = np.array(
        [
            [
                1057.66078,
                4802.57196,
                1298.08163,
                1217.39827,
                456.70756,
                13143.74176,
                8244.52203,
                21364.59699,
                877.14343,
                207.58535,
            ]
        ]
    ).T
    return var
