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
"""Monte Carlo iterator."""

import logging

import matplotlib.pyplot as plt
import numpy as np

from queens.iterators.sequence_iterator import SequenceIterator
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class MonteCarloIterator(SequenceIterator):
    """Basic Monte Carlo Iterator to enable MC sampling.

    Attributes:
        num_samples (int): Number of samples to compute.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        seed,
        num_samples,
        result_description=None,
    ):
        """Initialise Monte Carlo iterator.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            seed  (int):                Seed for random number generation
            num_samples (int):          Number of samples to compute
            result_description (dict, opt):  Description of desired results
        """
        super().__init__(model, parameters, global_settings, seed, result_description)
        self.num_samples = num_samples

    def generate_inputs(self):
        np.random.seed(self.seed)
        return self.parameters.draw_samples(self.num_samples)

    def post_run(self):
        """Analyze the results."""
        super().post_run()

        if self.result_description is not None:
            # ----------------------------- WIP PLOT OPTIONS ----------------------------
            if self.result_description.get("plot_results", False):
                _, ax = plt.subplots(figsize=(6, 4))

                # Check for dimensionality of the results
                if self.outputs["result"].shape[1] == 1:
                    ax.hist(self.outputs["result"], bins="auto")
                    ax.set_xlabel(r"Output")
                    ax.set_ylabel(r"Count [-]")
                    plt.tight_layout()
                    plt.savefig(self.global_settings.result_file(".png"))
                    plt.show()
                else:
                    _logger.warning(
                        "Plotting is not implemented yet for a multi-dimensional model output"
                    )
