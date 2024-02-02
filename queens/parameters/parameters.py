"""Parameters module."""

import logging

import numpy as np

from queens.distributions import VALID_TYPES as VALID_DISTRIBUTION_TYPES
from queens.distributions.distributions import ContinuousDistribution
from queens.parameters.fields import VALID_TYPES as VALID_FIELD_TYPES
from queens.parameters.fields.random_fields import RandomField
from queens.utils.import_utils import get_module_class
from queens.utils.logger_settings import log_init_args

VALID_TYPES = VALID_DISTRIBUTION_TYPES | VALID_FIELD_TYPES

_logger = logging.getLogger(__name__)


def from_config_create_parameters(parameters_options, pre_processor=None):
    """Create a QUEENS parameter object from config.

    Args:
        parameters_options (dict): Parameters description
        pre_processor (obj, optional): Pre-processor object to read coordinates of random field
                                       discretization
    """
    joint_parameters_dict = {}
    for parameter_name, parameter_dict in parameters_options.items():
        parameter_class = get_module_class(parameter_dict, VALID_TYPES)
        if issubclass(parameter_class, ContinuousDistribution):
            parameter_object = parameter_class(**parameter_dict)
        elif issubclass(parameter_class, RandomField):
            parameter_object = parameter_class(
                **parameter_dict, coords=pre_processor.coords_dict[parameter_name]
            )
        else:
            raise NotImplementedError(f"Parameter type '{parameter_class.__name__}' not supported.")
        joint_parameters_dict[parameter_name] = parameter_object

    return Parameters(**joint_parameters_dict)


def _add_parameters_keys(parameters_keys, parameter_name, dimension):
    """Add parameter keys to existing parameter keys.

    If the dimension of a parameter is larger than one, a separate unique key is added for each
    parameter member.
    Example: If parameter x1 is 3-dimensional the keys x1_0, x1_1, x1_2 is added.
             If parameter x1 is 1-dimensional the key x1 is added.

    Args:
        parameters_keys (list): List of existing parameter keys
        parameter_name (str): Parameter name to be added
        dimension (int): dimension of Parameter

    Returns:
        parameters_keys (list): List of keys for all parameter members
    """
    if dimension == 1:
        parameters_keys.append(parameter_name)
    else:
        parameters_keys.extend([f"{parameter_name}_{i}" for i in range(dimension)])
    return parameters_keys


class Parameters:
    """Parameters class.

    Attributes:
        dict (dict): Random variables and random fields stored in a dict.
        parameters_keys (list): List of keys for all parameter members.
        num_parameters (int): Number of (truncated) parameters.
        random_field_flag (bool): Specifies if random fields are used.
        names (list): Parameter names.
    """

    @log_init_args
    def __init__(self, **parameters):
        """Initialize Parameters object.

        Args:
            **parameters (ContinuousDistribution, RandomField): parameters as keyword arguments
        """
        joint_parameters_keys = []
        joint_parameters_dim = 0
        random_field_flag = False

        for parameter_name, parameter_obj in parameters.items():
            if isinstance(parameter_obj, ContinuousDistribution):
                joint_parameters_keys = _add_parameters_keys(
                    joint_parameters_keys, parameter_name, parameter_obj.dimension
                )
                joint_parameters_dim += parameter_obj.dimension
            elif isinstance(parameter_obj, RandomField):
                joint_parameters_keys += parameter_obj.coords['keys']
                joint_parameters_dim += parameter_obj.dim_truncated
                random_field_flag = True
            else:
                raise NotImplementedError(
                    f"Parameter class '{parameter_obj.__class__.__name__}' " "not supported."
                )

        self.dict = parameters
        self.parameters_keys = joint_parameters_keys
        self.num_parameters = joint_parameters_dim
        self.random_field_flag = random_field_flag
        self.names = list(parameters.keys())

    def draw_samples(self, num_samples):
        """Draw samples from all parameters.

        Args:
            num_samples: TODO_doc
        Returns:
            samples (np.ndarray): Drawn samples
        """
        samples = np.zeros((num_samples, self.num_parameters))
        current_index = 0
        for parameter in self.to_list():
            samples[:, current_index : current_index + parameter.dimension] = parameter.draw(
                num_samples
            )
            current_index += parameter.dimension
        return samples

    def joint_logpdf(self, samples):
        """Evaluate the logpdf summed over all parameters.

        Args:
            samples: TODO_doc
        Returns:
            logpdf (np.ndarray): logpdf summed over all parameters
        """
        samples = samples.reshape(-1, self.num_parameters)
        logpdf = 0
        i = 0
        for parameter in self.to_list():
            logpdf += parameter.logpdf(samples[:, i : i + parameter.dimension])
            i += parameter.dimension
        return logpdf

    def grad_joint_logpdf(self, samples):
        """Evaluate the gradient of the joint logpdf w.r.t. the samples.

        Args:
            samples: TODO_doc
        Returns:
            grad_logpdf (np.ndarray): Gradient of the joint logpdf w.r.t. the samples
        """
        samples = samples.reshape(-1, self.num_parameters)
        grad_logpdf = np.zeros(samples.shape)
        j = 0
        for parameter in self.to_list():
            grad_logpdf[:, j : j + parameter.dimension] = parameter.grad_logpdf(
                samples[:, j : j + parameter.dimension]
            )
            j += parameter.dimension
        return grad_logpdf

    def inverse_cdf_transform(self, samples):
        """Transform samples to unit interval.

        Args:
            samples (np.ndarray): Samples that should be transformed.

        Returns:
            transformed_samples (np.ndarray): Transformed samples
        """
        samples = samples.reshape(-1, self.num_parameters)
        transformed_samples = np.zeros(samples.shape)
        for i, parameter in enumerate(self.to_list()):
            if parameter.dimension != 1:
                raise ValueError("Only 1D Random variables can be transformed!")
            transformed_samples[:, i] = parameter.ppf(samples[:, i])
        return transformed_samples

    def sample_as_dict(self, sample):
        """Return sample as a dict.

        Args:
            sample (np.ndarray): A single sample

        Returns:
            sample_dict (dict): Dictionary containing sample members and the corresponding parameter
            keys
        """
        sample_dict = {}
        sample = sample.reshape(-1)
        if self.random_field_flag:
            sample = self.expand_random_field_realization(sample)
        for j, key in enumerate(self.parameters_keys):
            sample_dict[key] = sample[j]
        return sample_dict

    def expand_random_field_realization(self, truncated_sample):
        """Expand truncated representation of random fields.

        Args:
            truncated_sample (np.ndarray): Truncated representation of sample

        Returns:
            sample_expanded (np.ndarray): Expanded representation of sample
        """
        sample_expanded = np.zeros(len(self.parameters_keys))
        index_truncated = 0
        index_expanded = 0
        for parameter in self.to_list():
            if isinstance(parameter, RandomField):
                sample_expanded[
                    index_expanded : index_expanded + parameter.dimension
                ] = parameter.expanded_representation(
                    truncated_sample[index_truncated : index_truncated + parameter.dim_truncated]
                )
                index_expanded += parameter.dimension
                index_truncated += parameter.dim_truncated
            else:
                sample_expanded[
                    index_expanded : index_expanded + parameter.dimension
                ] = truncated_sample[index_truncated : index_truncated + parameter.dimension]
                index_expanded += parameter.dimension
                index_truncated += parameter.dimension
        return sample_expanded

    def to_list(self):
        """Return parameters as list.

        Returns:
            parameter_list (list): List of parameters
        """
        parameter_list = list(self.dict.values())
        return parameter_list

    def latent_grad(self, upstream_gradient):
        """Gradient of the rvs and rfs w.r.t. latent variables.

        Args:
            upstream_gradient (np.array): Upstream gradient
        Returns:
            gradient (np.ndarray): Gradient of the joint rvs/rfs w.r.t. the samples
        """
        return upstream_gradient

    def to_distribution_list(self):
        """Return the distributions of the parameters as list.

        Returns:
            parameter_list (list): List of distributions of parameters
        """
        return list(self.dict.values())