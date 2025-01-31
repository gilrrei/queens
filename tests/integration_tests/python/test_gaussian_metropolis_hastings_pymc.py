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
"""Integration test for the Metropolis Hastings PyMC iterator."""

import numpy as np
import pytest
from mock import patch

from queens.distributions.normal import NormalDistribution
from queens.drivers.function_driver import FunctionDriver
from queens.example_simulator_functions.gaussian_logpdf import gaussian_2d_logpdf
from queens.iterators.metropolis_hastings_pymc_iterator import MetropolisHastingsPyMCIterator
from queens.main import run_iterator
from queens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.experimental_data_reader import ExperimentalDataReader
from queens.utils.io_utils import load_result


def test_gaussian_mh(tmp_path, _create_experimental_data_zero, global_settings):
    """Test case for mh iterator."""
    # Parameters
    x1 = NormalDistribution(mean=[-2.0, 2.0], covariance=[[1.0, 0.0], [0.0, 1.0]])
    parameters = Parameters(x1=x1)

    # Setup iterator
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=tmp_path,
        output_label="y_obs",
    )
    driver = FunctionDriver(parameters=parameters, function="patch_for_likelihood")
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)
    forward_model = SimulationModel(scheduler=scheduler, driver=driver)
    model = GaussianLikelihood(
        noise_type="fixed_variance",
        noise_value=1.0,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = MetropolisHastingsPyMCIterator(
        seed=42,
        num_samples=10,
        num_burn_in=2,
        num_chains=1,
        use_queens_prior=False,
        progressbar=False,
        result_description={"write_results": True, "plot_results": False, "cov": True},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    with patch.object(GaussianLikelihood, "evaluate", target_density):
        run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    assert results["mean"].mean(axis=0) == pytest.approx(
        np.array([-0.5680310153118374, 0.9247536392514567])
    )
    assert results["variance"].mean(axis=0) == pytest.approx(
        [0.13601070852470507, 0.6672200465857734]
    )


def target_density(self, samples):  # pylint: disable=unused-argument
    """Patch likelihood."""
    samples = np.atleast_2d(samples)
    log_likelihood = gaussian_2d_logpdf(samples).flatten()

    return log_likelihood
