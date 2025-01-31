#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Integration test for the Gaussian Bayesian neural network model.

This test uses a Monte Carlo simulation.
"""

import numpy as np
import pytest

from queens.distributions.uniform import UniformDistribution
from queens.drivers.function_driver import FunctionDriver
from queens.iterators.monte_carlo_iterator import MonteCarloIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.models.surrogate_models.bayesian_neural_network import (
    GaussianBayesianNeuralNetworkModel,
)
from queens.parameters.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.io_utils import load_result
from test_utils.integration_tests import assert_monte_carlo_iterator_results


def test_neural_network_gauss_bayesian_branin(expected_mean, expected_var, global_settings):
    """Test case for Bayesian neural network model."""
    # Parameters
    x1 = UniformDistribution(lower_bound=-5, upper_bound=10)
    x2 = UniformDistribution(lower_bound=0, upper_bound=15)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    driver = FunctionDriver(parameters=parameters, function="branin78_hifi")
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)
    simulation_model = SimulationModel(scheduler=scheduler, driver=driver)
    training_iterator = MonteCarloIterator(
        seed=42,
        num_samples=100,
        result_description=None,
        model=simulation_model,
        parameters=parameters,
        global_settings=global_settings,
    )
    model = GaussianBayesianNeuralNetworkModel(
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
        num_samples_statistics=10,
        num_epochs=100,
        adams_training_rate=0.1,
        optimizer_seed=1,
        nodes_per_hidden_layer_lst=[10],
        activation_per_hidden_layer_lst=["sigmoid"],
        verbosity_on=True,
        training_iterator=training_iterator,
    )
    iterator = MonteCarloIterator(
        seed=44,
        num_samples=10,
        result_description={"write_results": True},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))
    assert_monte_carlo_iterator_results(results["outputs"], expected_mean, expected_var)


@pytest.fixture(name="expected_mean")
def fixture_expected_mean():
    """Expected mean values."""
    mean = np.array(
        [
            [
                65.37786,
                65.44934,
                44.39922,
                57.19025,
                64.86770,
                65.44933,
                65.44935,
                65.44935,
                65.44862,
                22.31277,
            ]
        ]
    )
    return mean.T


@pytest.fixture(name="expected_var")
def fixture_expected_var():
    """Expected variance values."""
    var = np.array(
        [
            [
                3.31274,
                3.31792,
                2.04469,
                2.76017,
                3.27650,
                3.31792,
                3.31792,
                3.31792,
                3.31787,
                1.08863,
            ]
        ]
    )
    return var.T
