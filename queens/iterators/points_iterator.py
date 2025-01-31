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
"""Iterator to run a model on predefined input points."""

import logging

import numpy as np

from queens.iterators.sequence_iterator import SequenceIterator
from queens.utils.ascii_art import print_points_iterator
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class PointsIterator(SequenceIterator):
    """Iterator at given input points.

    Attributes:
        result_description (dict): Settings for storing
        output (np.array): Array with all model outputs
        points (dict): Dictionary with name and samples
        points_array (np.ndarray): Array with all samples
    """

    @log_init_args
    def __init__(self, model, parameters, global_settings, points, result_description):
        """Initialise Iterator.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            points (dict): Dictionary with name and samples
            result_description (dict): Settings for storing
        """
        super().__init__(model, parameters, global_settings, result_description, None)
        self.points = points
        self.result_description = result_description

    def generate_inputs(self):
        """Prerun."""
        print_points_iterator()
        _logger.info("Creating points.")

        points = []
        for name in self.parameters.names:
            points.append(np.array(self.points[name]))

        # number of points for each parameter
        points_lengths = [len(d) for d in points]

        # check if the provided number of points is equal for each parameter
        if len(set(points_lengths)) != 1:
            message = ", ".join(
                [f"{n}: {l}" for n, l in zip(self.parameters.names, points_lengths)]
            )
            raise ValueError(
                "Non-matching number of points for the different parameters: " + message
            )
        return np.array(points).T
