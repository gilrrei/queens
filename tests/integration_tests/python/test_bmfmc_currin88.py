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
"""Integration tests for the BMFMC routine.

The test is based on the high-fidelity Currin function.
"""

# pylint: disable=invalid-name
import numpy as np
from scipy.stats import entropy

from queens.distributions.uniform import UniformDistribution
from queens.drivers.function_driver import FunctionDriver
from queens.iterators.bmfmc_iterator import BMFMCIterator
from queens.main import run_iterator
from queens.models.bmfmc_model import BMFMCModel
from queens.models.simulation_model import SimulationModel
from queens.models.surrogate_models.gp_approximation_gpflow import GPFlowRegressionModel
from queens.parameters.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.io_utils import load_result
from queens.utils.pdf_estimation import estimate_pdf


# ---- actual integration tests -------------------------------------------------
def test_bmfmc_iterator_currin88_random_vars_diverse_design(
    tmp_path,
    _write_lf_mc_data_to_pickle,
    hf_mc_data,
    bandwidth_lf_mc,
    design_method,
    global_settings,
):
    """Integration tests for BMFMC routine using *currin88* function.

    The test uses a high-fidelity (HF) and a low-fidelity (LF) version
    of the *currin88* function.
    """
    plot_dir = tmp_path
    lf_mc_data_name = "LF_MC_data.pickle"
    path_lf_mc_pickle_file = tmp_path / lf_mc_data_name
    # Parameters
    x1 = UniformDistribution(lower_bound=0.0, upper_bound=1.0)
    x2 = UniformDistribution(lower_bound=0.0, upper_bound=1.0)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    probabilistic_mapping = GPFlowRegressionModel(
        train_likelihood_variance=False,
        number_restarts=2,
        number_training_iterations=1000,
        dimension_lengthscales=2,
    )
    driver = FunctionDriver(parameters=parameters, function="currin88_hifi")
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)
    hf_model = SimulationModel(scheduler=scheduler, driver=driver)
    model = BMFMCModel(
        predictive_var=False,
        BMFMC_reference=False,
        y_pdf_support_min=-0.5,
        y_pdf_support_max=15.0,
        path_to_lf_mc_data=(path_lf_mc_pickle_file,),
        path_to_hf_mc_reference_data=None,
        features_config="opt_features",
        num_features=1,
        probabilistic_mapping=probabilistic_mapping,
        hf_model=hf_model,
        parameters=parameters,
        global_settings=global_settings,
    )
    iterator = BMFMCIterator(
        global_settings=global_settings,
        result_description={
            "write_results": True,
            "plotting_options": {
                "plot_booleans": [False, False, False],
                "plotting_dir": plot_dir,
                "plot_names": ["pdfs.eps", "manifold.eps", "ranking.eps"],
                "save_bool": [False, False, False],
                "animation_bool": False,
            },
        },
        initial_design={
            "num_HF_eval": 100,
            "num_bins": 50,
            "method": design_method,
            "seed": 1,
            "master_LF": 0,
        },
        model=model,
        parameters=parameters,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    # get the y_support and calculate HF MC reference
    y_pdf_support = results["outputs"]["y_pdf_support"]

    p_yhf_mc, _ = estimate_pdf(
        np.atleast_2d(hf_mc_data).T, bandwidth_lf_mc, support_points=np.atleast_2d(y_pdf_support)
    )

    kl_divergence = entropy(p_yhf_mc, results["outputs"]["p_yhf_mean"])
    assert kl_divergence < 0.3
