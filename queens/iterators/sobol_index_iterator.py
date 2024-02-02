"""Estimate Sobol indices."""

import logging

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from SALib.analyze import sobol
from SALib.sample import saltelli

from queens.distributions import lognormal, normal, uniform
from queens.iterators.iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results

_logger = logging.getLogger(__name__)


# TODO deal with non-uniform input distribution
class SobolIndexIterator(Iterator):
    """Sobol Index Iterator.

    This class essentially provides a wrapper around the SALib library.

    Attributes:
        seed (int): Seed for random number generator.
        num_samples (int): Number of samples.
        calc_second_order (bool): Calculate second-order sensitivities.
        num_bootstrap_samples (int): Number of bootstrap samples.
        confidence_level (float): The confidence interval level.
        result_description (dict): TODO_doc
        samples (np.array): Array with all samples.
        output (dict): Dict with all outputs corresponding to
                       samples.
        salib_problem (dict): Problem definition for SALib.
        num_params (int): Number of parameters.
        parameter_names (list): List with parameter names.
        sensitivity_indices (dict): Dictionary with sensitivity indices.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        seed,
        num_samples,
        calc_second_order,
        num_bootstrap_samples,
        confidence_level,
        result_description,
    ):
        """Initialize Saltelli SALib iterator object.

        Args:
            model (model): Model to be evaluated by iterator
            parameters (obj): Parameters object
            seed (int): Seed for random number generation
            num_samples (int): Number of desired (random) samples
            calc_second_order (bool): Calculate second-order sensitivities
            num_bootstrap_samples (int): Number of bootstrap samples
            confidence_level (float): The confidence interval level
            result_description (dict): Dictionary with desired result description
        """
        super().__init__(model, parameters)

        self.seed = seed
        self.num_samples = num_samples
        self.calc_second_order = calc_second_order
        self.num_bootstrap_samples = num_bootstrap_samples
        self.confidence_level = confidence_level
        self.result_description = result_description

        self.samples = None
        self.output = None
        self.salib_problem = None
        self.num_params = self.parameters.num_parameters
        self.parameter_names = []
        self.sensitivity_indices = None

    def pre_run(self):
        """Generate samples for subsequent analysis and update model."""
        # setup SALib problem dict
        bounds = []
        dists = []
        for key, parameter in self.parameters.dict.items():
            self.parameter_names.append(key)
            if isinstance(parameter, uniform.UniformDistribution):
                upper_bound = parameter.upper_bound
                lower_bound = parameter.lower_bound
                distribution_name = 'unif'
            # in queens normal distributions are parameterized with mean and var
            # in salib normal distributions are parameterized via mean and std
            # -> we need to reparameterize normal distributions
            elif isinstance(parameter, normal.NormalDistribution):
                lower_bound = parameter.mean.squeeze()
                upper_bound = np.sqrt(parameter.covariance.squeeze())
                distribution_name = 'norm'
            elif isinstance(parameter, lognormal.LogNormalDistribution):
                lower_bound = parameter.mu.squeeze()
                upper_bound = parameter.sigma.squeeze()
                distribution_name = 'lognorm'
            else:
                raise ValueError("Valid distributions are normal, lognormal and uniform!")

            bounds.append([lower_bound, upper_bound])
            dists.append(distribution_name)

        if self.parameters.random_field_flag:
            raise RuntimeError(
                "The SaltelliIterator does not work in conjunction with random fields."
            )

        self.salib_problem = {
            'num_vars': self.parameters.num_parameters,
            'names': self.parameters.names,
            'bounds': bounds,
            'dists': dists,
        }

        _logger.info("Draw %s samples...", self.num_samples)
        self.samples = saltelli.sample(
            self.salib_problem,
            self.num_samples,
            calc_second_order=self.calc_second_order,
            skip_values=1024,
        )
        _logger.debug(self.samples)

    def get_all_samples(self):
        """Return all samples."""
        return self.samples

    def core_run(self):
        """Run Analysis on model."""
        _logger.info("Evaluate model...")
        self.output = self.model.evaluate(self.samples)

        _logger.info("Calculate Sensitivity Indices...")
        self.sensitivity_indices = sobol.analyze(
            self.salib_problem,
            np.reshape(self.output['result'], (-1)),
            calc_second_order=self.calc_second_order,
            num_resamples=self.num_bootstrap_samples,
            conf_level=self.confidence_level,
            print_to_console=False,
            seed=self.seed,
        )

    def post_run(self):
        """Analyze the results."""
        results = self.process_results()
        if self.result_description is not None:
            if self.result_description["write_results"] is True:
                write_results(results, self.output_dir, self.experiment_name)
            self.print_results(results)
            if self.result_description["plot_results"] is True:
                self.plot_results(results)

    def print_results(self, results):
        """Print results.

        Args:
            results (dict): Dictionary with Sobol indices and confidence intervals
        """
        S = results["sensitivity_indices"]
        parameter_names = results["parameter_names"]

        additivity = np.sum(S["S1"])
        higher_interactions = 1 - additivity
        S_df = pd.DataFrame(
            {key: value for (key, value) in S.items() if key not in ["S2", "S2_conf"]},
            index=parameter_names,
        )
        _logger.info("Main and Total Effects:")
        _logger.info(S_df)
        _logger.info("Additivity, sum of main effects (for independent variables):")
        _logger.info('S_i = %s', additivity)

        if self.calc_second_order:
            S2 = S["S2"]
            S2_conf = S["S2_conf"]

            for j in range(self.parameters.num_parameters):
                S2[j, j] = S["S1"][j]
                for k in range(j + 1, self.parameters.num_parameters):
                    S2[k, j] = S2[j, k]
                    S2_conf[k, j] = S2_conf[j, k]

            S2_df = pd.DataFrame(S2, columns=parameter_names, index=parameter_names)
            S2_conf_df = pd.DataFrame(S2_conf, columns=parameter_names, index=parameter_names)

            _logger.info("Second Order Indices (diagonal entries are main effects):")
            _logger.info(S2_df)
            _logger.info("Confidence Second Order Indices:")
            _logger.info(S2_conf_df)

            # we extract the upper triangular matrix which includes the diagonal entries
            # therefore we have to subtract the trace
            second_order_interactions = np.sum(np.triu(S2, k=1))
            higher_interactions = higher_interactions - second_order_interactions
            str_second_order_interactions = f'S_ij = {second_order_interactions}'

            _logger.info("Sum of second order interactions:")
            _logger.info(str_second_order_interactions)

            str_higher_order_interactions = f'1 - S_i - S_ij = {higher_interactions}'
        else:
            str_higher_order_interactions = f'1 - S_i = {higher_interactions}'

        _logger.info("Higher order interactions:")
        _logger.info(str_higher_order_interactions)

    def process_results(self):
        """Write all results to self contained dictionary.

        Returns:
            results (dict): Dictionary with Sobol indices and confidence intervals
        """
        results = {
            "parameter_names": self.parameters.names,
            "sensitivity_indices": self.sensitivity_indices,
            "second_order": self.calc_second_order,
            "samples": self.samples,
            "output": self.output,
        }

        return results

    def plot_results(self, results):
        """Create bar graph of first order sensitivity indices.

        Args:
            results (dict): Dictionary with Sobol indices and confidence intervals
        """
        experiment_name = self.experiment_name

        # Plot first-order indices also called main effect
        chart_name = experiment_name + '_S1.html'
        chart_path = self.output_dir / chart_name
        bars = go.Bar(
            x=results["parameter_names"],
            y=results["sensitivity_indices"]["S1"],
            error_y={
                "type": 'data',
                "array": results['sensitivity_indices']['S1_conf'],
                "visible": True,
            },
        )
        data = [bars]

        layout = {
            "title": 'First-Order Sensitivity Indices',
            "xaxis": {"title": 'Parameter'},
            "yaxis": {"title": 'Main Effect'},
        }

        fig = go.Figure(data=data, layout=layout)
        fig.write_html(chart_path)

        # Plot total indices also called total effect
        chart_name = experiment_name + '_ST.html'
        chart_path = self.output_dir / chart_name
        bars = go.Bar(
            x=results["parameter_names"],
            y=results["sensitivity_indices"]["ST"],
            error_y={
                "type": 'data',
                "array": results['sensitivity_indices']['ST_conf'],
                "visible": True,
            },
        )
        data = [bars]

        layout = {
            "title": 'Total Sensitivity Indices',
            "xaxis": {"title": 'Parameter'},
            "yaxis": {"title": 'Total Effect'},
        }

        fig = go.Figure(data=data, layout=layout)
        fig.write_html(chart_path)

        # Plot second order indices (if applicable)
        if self.calc_second_order:
            S2 = results["sensitivity_indices"]["S2"]
            S2 = S2[np.triu_indices(self.num_params, k=1)]

            S2_conf = results["sensitivity_indices"]["S2_conf"]
            S2_conf = S2_conf[np.triu_indices(self.num_params, k=1)]

            # build list of names of second-order indices
            names = []
            for i in range(1, self.num_params + 1):
                for j in range(i + 1, self.num_params + 1):
                    names.append(f"S{i}{j}")

            chart_name = experiment_name + '_S2.html'
            chart_path = self.output_dir / chart_name
            bars = go.Bar(
                x=names, y=S2, error_y={"type": 'data', "array": S2_conf, "visible": True}
            )
            data = [bars]

            layout = {
                "title": 'Second Order Sensitivity Indices',
                "xaxis": {"title": 'Parameter'},
                "yaxis": {"title": 'Second Order Effects'},
            }

            fig = go.Figure(data=data, layout=layout)
            fig.write_html(chart_path)