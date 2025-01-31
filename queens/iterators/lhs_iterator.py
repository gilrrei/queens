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
"""Latin hypercube sampling iterator."""

import logging

import numpy as np
from pyDOE import lhs

from queens.iterators.sequence_iterator import SequenceIterator
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class LHSIterator(SequenceIterator):
    """Basic LHS Iterator to enable Latin Hypercube sampling.

    Attributes:
        seed (int): Seed for numpy random number generator.
        num_samples (int):    Number of samples to compute.
        num_iterations (int): Number of optimization iterations of design.
        result_description (dict):  Description of desired results.
        criterion (str): Allowable values are:

            *   *center* or *c*
            *   *maximin* or *m*
            *   *centermaximin* or *cm*
            *   *correlation* or *corr*
        samples (np.array):   Array with all samples.
        output (np.array):   Array with all model outputs.
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
        num_iterations=10,
        criterion="maximin",
    ):
        """Initialise LHSiterator.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            seed (int): Seed for numpy random number generator
            num_samples (int):    Number of samples to compute
            result_description (dict, opt):  Description of desired results
            num_iterations (int): Number of optimization iterations of design
            criterion (str): Allowable values are "center" or "c", "maximin" or "m",
                             "centermaximin" or "cm", and "correlation" or "corr"
        """
        super().__init__(model, parameters, global_settings, seed, result_description)
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.criterion = criterion

    def generate_inputs(self):
        """Generate samples for subsequent LHS analysis."""
        np.random.seed(self.seed)

        num_inputs = self.parameters.num_parameters

        # create latin hyper cube samples in unit hyper cube
        hypercube_samples = lhs(
            num_inputs, self.num_samples, criterion=self.criterion, iterations=self.num_iterations
        )
        # scale and transform samples according to the inverse cdf
        return self.parameters.inverse_cdf_transform(hypercube_samples)
