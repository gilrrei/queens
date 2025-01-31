from queens.iterators.iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.sampler import get_samples_statistics
from queens.utils.process_outputs import write_results
from queens.utils.print_utils import get_str_table
import logging
import abc

_logger = logging.getLogger(__name__)


class SequenceIterator(Iterator):
    """Basic Monte Carlo Iterator to enable MC sampling.

    Attributes:
        seed  (int): Seed for random number generation.
        num_samples (int): Number of samples to compute.
        result_description (dict):  Description of desired results.
        samples (np.array):         Array with all samples.
        output (np.array):          Array with all model outputs.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        seed,
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
        super().__init__(model, parameters, global_settings)
        self.seed = seed
        self.result_description = result_description
        self.inputs = None
        self.outputs = None

    @abc.abstractmethod
    def generate_inputs(self):
        pass

    def pre_run(self):
        self.inputs = self.generate_inputs()
        _logger.info(get_str_table("Input samples.", get_samples_statistics(self.inputs)))
        _logger.debug("Inputs %s", self.inputs)

    def core_run(self):
        self.outputs = self.model.evaluate(self.inputs)
        _logger.debug("Outputs %s", self.outputs["result"])

    def post_run(self):
        if self.result_description is not None:
            if self.result_description["write_results"]:
                results = self.get_results()
                write_results(results, self.global_settings.result_file(".pickle"))

    def get_results(self):
        results = get_samples_statistics(self.outputs["result"])
        _logger.info(get_str_table("Output samples.", results))
        results["inputs"] = self.inputs
        results["outputs"] = self.outputs
        return results
