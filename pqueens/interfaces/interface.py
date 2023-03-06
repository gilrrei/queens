"""Interface class to map input variables to simulation outputs."""
import abc

import numpy as np

import pqueens.parameters.parameters as parameters_module


class Interface(metaclass=abc.ABCMeta):
    """Interface class to map input variables to simulation outputs.

    The interface is responsible for the actual mapping between input
    variables and simulation outputs. The purpose of this base class is
    to define a unified interface on the one hand, while at the other
    hand taking care of the construction of the appropriate objects from
    the derived class.

    Attributes:
        name (str): Name of the interface.
        parameters (obj): Parameters object.
        latest_job_id (int):    Latest job ID.
    """

    def __init__(self, name):
        """Initialize interface object.

        Args:
            name (obj): Name of the interface.
        """
        self.name = name
        self.parameters = parameters_module.parameters
        self.latest_job_id = 0

    @abc.abstractmethod
    def evaluate(self, samples):
        """Evaluate samples.

        Mapping function which orchestrates call to external simulation
        software or approximation.

        Args:
            samples (list):  List of variables objects
        """

    def create_samples_list(self, samples):
        """Create a list of sample dictionaries.

        Args:
            samples (np.array): Samples of simulation input variables

        Returns:
            samples_list (list): List of dicts containing samples and job ids
        """
        number_of_samples = len(samples)

        # List of global sample ids
        sample_ids = np.arange(self.latest_job_id, self.latest_job_id + number_of_samples) + 1

        # Update the latest job id
        self.latest_job_id = int(self.latest_job_id + number_of_samples)

        # Create samples list and add job_id to the dicts
        samples_list = []
        for job_id, sample in zip(sample_ids, samples):
            sample_dict = self.parameters.sample_as_dict(sample)
            sample_dict.update({"job_id": int(job_id)})
            samples_list.append(sample_dict)

        return samples_list
