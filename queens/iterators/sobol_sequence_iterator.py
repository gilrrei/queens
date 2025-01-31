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
"""Sobol sequence iterator."""

import logging

from queens.iterators.sequence_iterator import SequenceIterator
from queens.utils.logger_settings import log_init_args
from queens.utils.sobol_sequence import sample_sobol_sequence

_logger = logging.getLogger(__name__)


class SobolSequenceIterator(SequenceIterator):
    """Sobol sequence in multiple dimensions.

    Attributes:
        seed  (int): This is the seed for the scrambling. The seed of the random number generator is
                     set to this, if specified. Otherwise, it uses a random seed.
        number_of_samples (int): Number of samples to compute.
        randomize (bool): Setting this to *True* will produce scrambled Sobol sequences. Scrambling
                          is capable of producing better Sobol sequences.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        seed,
        number_of_samples,
        result_description,
        randomize=False,
    ):
        """Initialize Sobol sequence iterator.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            seed  (int): This is the seed for the scrambling. The seed of the random number
                         generator is set to this, if specified. Otherwise, it uses a random seed.
            number_of_samples (int): Number of samples to compute
            result_description (dict):  Description of desired results
            randomize (bool): Setting this to True will produce scrambled Sobol sequences.
                              Scrambling is capable of producing better Sobol sequences.
        """
        super().__init__(model, parameters, global_settings, result_description, seed)
        self.number_of_samples = number_of_samples
        self.randomize = randomize

    def generate_inputs(self):
        """Generate samples for subsequent Sobol sequence analysis."""
        return sample_sobol_sequence(
            dimension=self.parameters.num_parameters,
            number_of_samples=self.number_of_samples,
            parameters=self.parameters,
            randomize=self.randomize,
            seed=self.seed,
        )
