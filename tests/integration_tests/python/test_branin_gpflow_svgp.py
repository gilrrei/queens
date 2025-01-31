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
"""Integration test for the GPflow based SVGP model."""

import numpy as np
import pytest

from queens.distributions.uniform import UniformDistribution
from queens.drivers.function_driver import FunctionDriver
from queens.iterators.monte_carlo_iterator import MonteCarloIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.models.surrogate_models.gp_approximation_gpflow_svgp import GPflowSVGPModel
from queens.parameters.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.io_utils import load_result
from test_utils.integration_tests import assert_monte_carlo_iterator_results


@pytest.mark.max_time_for_test(60)
def test_branin_gpflow_svgp(expected_mean, expected_var, global_settings):
    """Test case for GPflow based SVGP model."""
    # Parameters
    x1 = UniformDistribution(lower_bound=-5, upper_bound=10)
    x2 = UniformDistribution(lower_bound=0, upper_bound=15)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    driver = FunctionDriver(parameters=parameters, function="branin78_hifi")
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)
    model = SimulationModel(scheduler=scheduler, driver=driver)
    training_iterator = MonteCarloIterator(
        seed=42,
        num_samples=100,
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )
    model = GPflowSVGPModel(
        plotting_options={
            "plot_booleans": [False, False],
            "plotting_dir": "dummy",
            "plot_names": ["1D", "2D"],
            "save_bool": [False, False],
        },
        train_likelihood_variance=False,
        seed=41,
        mini_batch_size=50,
        number_inducing_points=50,
        train_inducing_points_location=True,
        number_training_iterations=10000,
        dimension_lengthscales=2,
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
            [181.62057979],
            [37.95455295],
            [47.86422341],
            [32.47391656],
            [23.99246991],
            [167.32578661],
            [106.07427664],
            [92.93591941],
            [50.72976800],
            [22.10505115],
        ]
    )
    return mean


@pytest.fixture(name="expected_var")
def fixture_expected_var():
    """Expected variance values."""
    var = np.array(
        [
            [4.62061],
            [1.38456],
            [0.96146],
            [0.20286],
            [0.34231],
            [1.03465],
            [0.24111],
            [0.40275],
            [0.22169],
            [0.58071],
        ]
    )
    return var
