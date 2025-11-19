#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
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
"""Driver to run a jobscript."""


import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from queens.drivers._driver import Driver
from queens.utils.exceptions import SubprocessError
from queens.utils.injector import inject, inject_in_template
from queens.utils.io import read_file
from queens.utils.logger_settings import log_init_args
from queens.utils.metadata import SimulationMetadata
from queens.utils.path import create_folder_if_not_existent
from queens.utils.run_subprocess import run_subprocess
from typing import Protocol

_logger = logging.getLogger(__name__)


class JobPreprocessor(Protocol):
    files_to_copy: list

    def prepare_input_files(
        self, sample_dict: dict, job_data: dict, experiment_dir: Path, job_dir: Path
    ) -> tuple[dict, Path, Path]: ...


class Injection(JobPreprocessor):

    def __init__(self, input_templates: dict, files_to_copy: list = None):
        if not isinstance(input_templates, dict):
            input_templates = {"input_file": input_templates}

        self.input_templates = {
            input_template_key: Path(input_template_path)
            for input_template_key, input_template_path in input_templates.items()
        }
        if files_to_copy is None:
            files_to_copy = []

        files_to_copy.extend((str(file) for file in self.input_templates.values()))

    def prepare_input_files(self, sample_dict, job_data, experiment_dir, job_dir):

        output_prefix = "output"
        output_dir = job_dir / output_prefix

        output_dir = create_folder_if_not_existent(output_dir)

        output_file = output_dir / output_prefix
        log_file = output_dir / (output_prefix + ".log")

        # Additional data
        job_data = job_data | {
            "output_dir": output_dir,
            "output_file": output_file,
            "log_file": log_file,
        }

        # Create the paths for all input files
        for input_template_name, input_template_path in self.input_templates.items():
            input_file_str = input_template_name + "".join(input_template_path.suffixes)
            job_data[input_template_name] = job_dir / input_file_str

        # Inject all the data into the all the input files
        for input_template_name, input_template_path in self.input_templates.items():
            inject(
                job_data | sample_dict,
                experiment_dir / input_template_path.name,
                job_data[input_template_name],
            )

        return job_data, output_dir, log_file


class Jobscript(Driver):
    """Driver to run an executable with a jobscript.

    Attributes:
        input_templates (Path): Read in simulation input template as string.
        data_processor (obj): Instance of data processor class.
        gradient_data_processor (obj): Instance of data processor class for gradient data.
        jobscript_template (str): Read-in jobscript template.
        jobscript_options (dict): Dictionary containing jobscript options.
        jobscript_file_name (str): Jobscript file name (default: 'jobscript.sh').
        raise_error_on_jobscript_failure (bool): Whether to raise an error for a non-zero jobscript
                                                 exit code.
    """

    @log_init_args
    def __init__(
        self,
        parameters,
        input_templates,
        jobscript_template,
        executable,
        files_to_copy=None,
        data_processor: Callable = None,
        gradient_data_processor: Callable = None,
        jobscript_file_name="jobscript.sh",
        extra_options=None,
        raise_error_on_jobscript_failure=True,
    ):
        """Initialize Jobscript object.

        Args:
            parameters (Parameters): Parameters object.
            input_templates (str, Path, dict): Path(s) to simulation input template.
            jobscript_template (str, Path): Path to jobscript template or read-in jobscript
                                            template.
            executable (str, Path): Path to main executable of respective software.
            files_to_copy (list, opt): Files or directories to copy to experiment_dir.
            data_processor (obj, opt): Instance of data processor class.
            gradient_data_processor (obj, opt): Instance of data processor class for gradient data.
            jobscript_file_name (str, opt): Jobscript file name (default: 'jobscript.sh').
            extra_options (dict, opt): Extra options to inject into jobscript template.
            raise_error_on_jobscript_failure (bool, opt): Whether to raise an error for a non-zero
                                                          jobscript exit code.
        """

        # preprocessing
        self.job_preprocessor: JobPreprocessor = Injection(input_templates, files_to_copy)

        super().__init__(parameters=parameters, files_to_copy=self.job_preprocessor.files_to_copy)

        # jobscript
        self.jobscript_template = self.get_read_in_jobscript_template(jobscript_template)

        if extra_options is None:
            extra_options = {}

        self.jobscript_options = extra_options
        self.jobscript_options["executable"] = executable
        self.jobscript_file_name = jobscript_file_name
        self.raise_error_on_jobscript_failure = raise_error_on_jobscript_failure

        # post processing
        self.data_processor = data_processor
        self.gradient_data_processor = gradient_data_processor

    @staticmethod
    def get_read_in_jobscript_template(jobscript_template):
        """Get the jobscript template contents.

        If the provided jobscript template is a Path or a string of a
        path and a valid file, the corresponding file is read.

        Args:
            jobscript_template (str, Path): Path to jobscript template or read-in jobscript
                                            template.

        Returns:
            str: Read-in jobscript template
        """
        if isinstance(jobscript_template, str):
            # Catch an exception due to a long string
            try:
                if Path(jobscript_template).is_file():
                    jobscript_template = read_file(jobscript_template)
            except OSError:
                _logger.debug(
                    "The provided jobscript template string is not a regular file so we assume "
                    "that it holds the read-in jobscript template. The jobscript template reads:\n"
                    "%s",
                    {jobscript_template},
                )

        elif isinstance(jobscript_template, Path):
            if jobscript_template.is_file():
                jobscript_template = read_file(jobscript_template)
            else:
                raise FileNotFoundError(
                    f"The provided jobscript template path {jobscript_template} is not a file."
                )
        else:
            raise TypeError("The jobscript template needs to be a string or a Path.")

        return jobscript_template

    def run(
        self,
        sample: np.ndarray,
        job_id: int,
        num_procs: int,
        experiment_dir: Path,
        experiment_name: str,
    ) -> dict:
        """Run the driver.

        Args:
            sample (np.array): Input sample.
            job_id (int): Job ID.
            num_procs (int): Number of processors.
            experiment_dir (Path): Path to QUEENS experiment directory.
            experiment_name (str): Name of QUEENS experiment.

        Returns:
            Result and potentially the gradient.
        """

        job_dir = experiment_dir / str(job_id)
        jobscript_file = job_dir / self.jobscript_file_name

        # Data for the jobs
        job_data = {
            "job_id": job_id,
            "num_procs": num_procs,
            "experiment_dir": experiment_dir,
            "experiment_name": experiment_name,
            "job_dir": job_dir,
        }

        sample_dict = self.parameters.sample_as_dict(sample)

        metadata = SimulationMetadata(job_id=job_id, inputs=sample_dict, job_dir=job_dir)

        with metadata.time_code("prepare_input_files"):

            # Create the input files with the samples
            job_data, output_dir, log_file = self.job_preprocessor.prepare_input_files(
                sample_dict, job_data, experiment_dir, job_dir
            )

            # Create jobscript
            inject_in_template(
                job_data | self.jobscript_options,
                self.jobscript_template,
                str(jobscript_file),
            )

        with metadata.time_code("run_jobscript"):
            execute_cmd = f"bash {jobscript_file} >{log_file} 2>&1"
            self._run_executable(job_id, execute_cmd)

        with metadata.time_code("data_processing"):
            results = self._get_results(output_dir)
            metadata.outputs = results

        return results

    def _run_executable(self, job_id, execute_cmd):
        """Run executable.

        Args:
            job_id (int): Job ID.
            execute_cmd (str): Executed command.
        """
        process_returncode, _, stdout, stderr = run_subprocess(
            execute_cmd,
            raise_error_on_subprocess_failure=False,
        )
        if self.raise_error_on_jobscript_failure and process_returncode:
            raise SubprocessError.construct_error_from_command(
                command=execute_cmd,
                command_output=stdout,
                error_message=stderr,
                additional_message=f"The jobscript with job ID {job_id} has failed with exit code "
                f"{process_returncode}.",
            )

    def _get_results(self, output_dir):
        """Get results from driver run.

        Args:
            output_dir (Path): Path to output directory.

        Returns:
            result (np.array): Result from the driver run.
            gradient (np.array, None): Gradient from the driver run (potentially None).
        """
        results = {}
        if self.data_processor:
            result = self.data_processor(output_dir)
            results["result"] = result
            _logger.debug("Got result: %s", result)

        if self.gradient_data_processor:
            gradient = self.gradient_data_processor(output_dir)
            results["gradient"] = gradient
            _logger.debug("Got gradient: %s", gradient)
        return results
