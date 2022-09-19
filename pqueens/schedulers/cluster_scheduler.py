"""Cluster scheduler for QUEENS runs."""
import atexit
import logging
import pathlib

import numpy as np

from pqueens.drivers import from_config_create_driver
from pqueens.utils.cluster_utils import get_cluster_job_id
from pqueens.utils.information_output import print_scheduling_information
from pqueens.utils.manage_singularity import SingularityManager
from pqueens.utils.path_utils import relative_path_from_pqueens
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.script_generator import generate_submission_script

from .scheduler import Scheduler

_logger = logging.getLogger(__name__)


class ClusterScheduler(Scheduler):
    """Cluster scheduler (either based on Slurm or Torque/PBS) for QUEENS.

    Attributes:
        port (int):                (only for remote scheduling with Singularity) port of
                                   remote resource for ssh port-forwarding to database
        remote (bool):             flag for remote scheduling
        remote connect (str):      (only for remote scheduling) address of remote
                                   computing resource
        singularity_manager (obj): instance of Singularity-manager class
    """

    def __init__(
        self,
        experiment_name,
        input_file,
        experiment_dir,
        driver_name,
        config,
        cluster_options,
        singularity,
        scheduler_type,
        singularity_manager,
        remote,
        remote_connect,
    ):
        """Init method for the cluster scheduler.

        Args:
            experiment_name (str):     name of QUEENS experiment
            input_file (path):         path to QUEENS input file
            experiment_dir (path):     path to QUEENS experiment directory
            driver_name (str):         Name of the driver that shall be used for job submission
            config (dict):             dictionary containing configuration as provided in
                                       QUEENS input file
            cluster_options (dict):    (only for cluster schedulers Slurm and PBS) further
                                       cluster options
            singularity (bool):        flag for use of Singularity containers
            scheduler_type (str):      type of scheduler chosen in QUEENS input file
            singularity_manager (obj): instance of Singularity-manager class
            remote (bool):             flag for remote scheduling
            remote_connect (str):      (only for remote scheduling) address of remote
                                       computing resource
        """
        super().__init__(
            experiment_name,
            input_file,
            experiment_dir,
            driver_name,
            config,
            cluster_options,
            singularity,
            scheduler_type,
        )
        self.singularity_manager = singularity_manager
        self.remote = remote
        self.port = None
        self.remote_connect = remote_connect

        # Close the ssh ports at when exiting after the queens run
        atexit.register(self.post_run)

    @classmethod
    def from_config_create_scheduler(cls, config, scheduler_name=None, driver_name=None):
        """Create cluster scheduler (Slurm or Torque/PBS) class for QUEENS.

        Args:
            config (dict): QUEENS input dictionary
            scheduler_name (str): Name of the scheduler
            driver_name (str): Name of the driver

        Returns:
            instance of cluster scheduler class
        """
        if not scheduler_name:
            scheduler_name = "scheduler"
        scheduler_options = config[scheduler_name]

        experiment_name = config['global_settings']['experiment_name']
        experiment_dir = pathlib.Path(scheduler_options['experiment_dir'])
        input_file = pathlib.Path(config["input_file"])
        singularity = scheduler_options.get('singularity', False)
        if not isinstance(singularity, bool):
            raise TypeError(
                f"The option 'singularity' in the scheduler part of the input file has to be a"
                f" boolean, however you provided '{singularity}' which is of type "
                f"{type(singularity)} "
            )
        scheduler_type = scheduler_options["scheduler_type"]

        cluster_options = scheduler_options.get("cluster", {})
        remote = scheduler_options.get('remote', False)
        if remote:
            remote_connect = scheduler_options['remote']['connect']
        else:
            remote_connect = None

        if remote and not singularity:
            raise NotImplementedError(
                "The combination of 'remote: true' and 'singularity: false' in the "
                "'scheduler section' is not implemented! "
                "Abort..."
            )

        if singularity:
            singularity_manager = SingularityManager(
                remote=remote,
                remote_connect=remote_connect,
                singularity_bind=scheduler_options['singularity_settings']['cluster_bind'],
                singularity_path=pathlib.Path(
                    scheduler_options['singularity_settings']['cluster_path']
                ),
                input_file=input_file,
            )
        else:
            singularity_manager = None

        # This hurts my brain
        cluster_options['job_name'] = None
        cluster_options['CLUSTERSCRIPT'] = cluster_options.get('script', None)
        cluster_options['nposttasks'] = scheduler_options.get('num_procs_post', 1)

        # set cluster options required specifically for PBS or Slurm
        if scheduler_type == 'pbs':
            cluster_options['start_cmd'] = 'qsub'
            rel_path = 'utils/jobscript_pbs.sh'
            cluster_options['pbs_queue'] = cluster_options.get('pbs_queue', 'batch')
            cls._set_number_procs_pbs(cluster_options, scheduler_options)
        elif scheduler_type == "slurm":
            cluster_options['start_cmd'] = 'sbatch'
            rel_path = 'utils/jobscript_slurm.sh'
            cluster_options['slurm_ntasks'] = str(scheduler_options.get('num_procs', 1))
        else:
            raise ValueError("Know cluster scheduler types are pbs or slurm")

        abs_path = relative_path_from_pqueens(rel_path)
        cluster_options['jobscript_template'] = abs_path

        if singularity:
            singularity_path = pathlib.Path(
                scheduler_options['singularity_settings']['cluster_path']
            )
            cluster_options['singularity_path'] = str(singularity_path)
            cluster_options['EXE'] = str(singularity_path.joinpath('singularity_image.sif'))
            cluster_options['singularity_bind'] = scheduler_options['singularity_settings'][
                'cluster_bind'
            ]
            cluster_options['OUTPUTPREFIX'] = ''
            cluster_options['DATAPROCESSINGFLAG'] = 'true'
        else:
            cluster_options['singularity_path'] = None
            cluster_options['singularity_bind'] = None
            cluster_options['DATAPROCESSINGFLAG'] = 'false'

        # TODO move this to a different place
        # print out scheduling information
        print_scheduling_information(
            scheduler_type,
            remote,
            remote_connect,
            singularity,
        )
        return cls(
            experiment_name=experiment_name,
            input_file=input_file,
            experiment_dir=experiment_dir,
            driver_name=driver_name,
            config=config,
            cluster_options=cluster_options,
            singularity=singularity,
            scheduler_type=scheduler_type,
            singularity_manager=singularity_manager,
            remote=remote,
            remote_connect=remote_connect,
        )

    @classmethod
    def _set_number_procs_pbs(cls, cluster_options, scheduler_options):
        """Set number of procs and nodes."""
        num_procs = scheduler_options.get('num_procs', 1)
        ppn = scheduler_options.get('pbs_num_avail_ppn', 16)
        if num_procs <= ppn:
            cluster_options['pbs_nodes'] = '1'
            cluster_options['pbs_ppn'] = str(num_procs)
        else:
            num_nodes = np.ceil(num_procs / ppn)
            if num_procs % num_nodes == 0:
                cluster_options['pbs_ppn'] = str(num_procs / num_nodes)
            else:
                raise ValueError(
                    "Number of tasks not evenly distributable, as required for PBS scheduler!"
                )

    # ------------------- CHILD METHODS THAT MUST BE IMPLEMENTED ------------------
    def pre_run(self):
        """Pre-run routine for local and remote computing with Singularity.

        Do automatic port-forwarding and copying files/folders.
        """
        # pre-run routines required when using Singularity both local and remote
        if self.singularity is True:
            self.singularity_manager.check_singularity_system_vars()
            self.singularity_manager.prepare_singularity_files()

            # pre-run routines required when using Singularity remote only
            if self.remote:
                _, _, hostname, _ = run_subprocess('hostname -i')
                _, _, username, _ = run_subprocess('whoami')
                address_localhost = username.rstrip() + r'@' + hostname.rstrip()

                self.singularity_manager.kill_previous_queens_ssh_remote(username)
                self.singularity_manager.establish_port_forwarding_local(address_localhost)
                self.port = self.singularity_manager.establish_port_forwarding_remote(
                    address_localhost
                )

                self.singularity_manager.copy_temp_json()

    def _submit_singularity(self, job_id, batch):
        """Submit job remotely to Singularity.

        Args:
            job_id (int):    ID of job to submit
            batch (int):     Batch number of job

        Returns:
            int:            process ID
        """
        if self.remote:
            # set job name as well as paths to input file and
            # destination directory for jobscript
            self.cluster_options['job_name'] = f"{self.experiment_name}_{job_id}"
            self.cluster_options['INPUT'] = (
                f"--job_id={job_id} --batch={batch} --port={self.port} --path_json="
                + f"{self.cluster_options['singularity_path']} --driver_name="
                + f"{self.driver_name} --workdir"
            )

            self.cluster_options['DESTDIR'] = str(
                self.experiment_dir.joinpath(str(job_id), 'output')
            )

            # generate jobscript for submission
            submission_script_path = self.experiment_dir.joinpath('jobfile.sh')
            generate_submission_script(
                self.cluster_options,
                str(submission_script_path),
                self.cluster_options['jobscript_template'],
                self.remote_connect,
            )

            # submit subscript remotely
            cmdlist_remote_main = [
                'ssh',
                self.remote_connect,
                '"cd',
                str(self.experiment_dir),
                ';',
                self.cluster_options['start_cmd'],
                str(submission_script_path),
                '"',
            ]
            cmd_remote_main = ' '.join(cmdlist_remote_main)
            _, _, stdout, _ = run_subprocess(
                cmd_remote_main,
                additional_message_error="The file 'remote_main' in remote singularity image "
                "could not be executed properly!",
            )

            # check matching of job ID
            match = get_cluster_job_id(self.scheduler_type, stdout)

            try:
                return int(match)
            except ValueError:
                _logger.error(stdout)
                return None
        else:
            raise ValueError("\nSingularity cannot yet be used locally on computing clusters!")

    # TODO this method needs to be replaced by job_id/scheduler_id check
    #  we can only check here if job was completed but might still be failed though
    def check_job_completion(self, job):
        """Check whether this job has been completed.

        Args:
            job (dict): Job dict.

        Returns:
            completed (bool): If job is completed
            failed (bool): If job failed.
        """
        # initialize completion and failure flags to false
        # (Note that failure is not checked for cluster scheduler
        #  and returned false in any case.)
        completed = True
        failed = False
        job_id = job['id']

        if self.scheduler_type == 'pbs':
            check_cmd = 'qstat'
            check_loc = -2
        elif self.scheduler_type == 'slurm':
            check_cmd = 'squeue --job'
            check_loc = -4
        else:
            raise RuntimeError(
                f'Unknown scheduler type "{self.scheduler_type}"! Valid options are pbs, slurm'
            )

        if self.remote:
            # set check command, check location and delete command for PBS or SLURM
            # generate check command
            command_list = [
                'ssh',
                self.remote_connect,
                '"',
                check_cmd,
                str(self.process_ids[str(job_id)]),
                '"',
            ]
        else:
            # generate check command
            command_list = [
                check_cmd,
                str(self.process_ids[str(job_id)]),
            ]

        command_string = ' '.join(command_list)
        _, _, stdout, _ = run_subprocess(command_string)

        if stdout:
            # split output string
            output = stdout.split()

            # second/fourth to last entry should be job status
            status = output[check_loc]
            if status in ['Q', 'R', 'H', 'S']:
                completed = False

        return completed, failed

    def post_run(self):
        """Post-run routine.

        Only for remote computing with Singularity: close ports.
        """
        if self.remote and self.singularity:
            self.singularity_manager.close_local_port_forwarding()
            self.singularity_manager.close_remote_port(self.port)
            print('All port-forwardings were closed again.')

    def _submit_driver(self, job_id, batch):
        """Submit job to driver.

        Args:
            job_id (int):    ID of job to submit
            batch (int):     Batch number of job

        Returns:
            driver_obj.pid (int): process ID
        """
        # create driver
        # TODO we should not create the object here everytime!
        # TODO instead only update the attributes of the instance.
        # TODO we should specify the data base sheet as well
        driver_obj = from_config_create_driver(
            self.config, job_id, batch, self.driver_name, cluster_options=self.cluster_options
        )
        # run driver and get process ID
        driver_obj.pre_job_run_and_run_job()
        pid = driver_obj.pid

        return pid
