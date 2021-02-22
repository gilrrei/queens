""" Collect fixtures used by the integration tests. """

import getpass
import os
import pathlib

import pytest

from pqueens.utils.manage_singularity import SingularityManager
from pqueens.utils.run_subprocess import run_subprocess


@pytest.fixture(scope='session')
def inputdir():
    """ Return the path to the json input-files of the function test. """
    dirpath = os.path.dirname(__file__)
    input_files_path = os.path.join(dirpath, 'queens_input_files')
    return input_files_path


@pytest.fixture(scope='session')
def third_party_inputs():
    """ Return the path to the json input-files of the function test. """
    dirpath = os.path.dirname(__file__)
    input_files_path = os.path.join(dirpath, 'third_party_input_files')
    return input_files_path


@pytest.fixture(scope='session')
def config_dir():
    """ Return the path to the json input-files of the function test. """
    dirpath = os.path.dirname(__file__)
    config_dir_path = os.path.join(dirpath, '../../../config')
    return config_dir_path


@pytest.fixture()
def set_baci_links_for_gitlab_runner(config_dir):
    """ Set symbolic links for baci on testing machine. """
    dst_baci = os.path.join(config_dir, 'baci-release')
    dst_drt_monitor = os.path.join(config_dir, 'post_drt_monitor')
    home = pathlib.Path.home()
    src_baci = pathlib.Path.joinpath(home, 'workspace/build/baci-release')
    src_drt_monitor = pathlib.Path.joinpath(home, 'workspace/build/post_drt_monitor')
    return dst_baci, dst_drt_monitor, src_baci, src_drt_monitor


# CLUSTER TESTS ------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def user():
    """ Name of user calling the test suite. """
    return getpass.getuser()


@pytest.fixture(scope="session")
def cluster_user(user):
    """ Username of cluster account to use for tests. """
    # user who calles the test suite
    # gitlab-runner has to run simulation as different user on cluster everyone else should use
    # account with same name
    if user == "gitlab-runner":
        cluster_user = "queens"
    else:
        cluster_user = user
    return cluster_user


@pytest.fixture(scope="session", params=["deep", "bruteforce"])
def cluster(request):
    return request.param


@pytest.fixture(scope="session")
def cluster_address(cluster):
    """ String used for ssh connect to the cluster. """
    address = cluster + '.lnm.mw.tum.de'
    return address


@pytest.fixture(scope="session")
def connect_to_resource(cluster_user, cluster):
    """ String used for ssh connect to the cluster. """
    connect_to_resource = cluster_user + '@' + cluster + '.lnm.mw.tum.de'
    return connect_to_resource


@pytest.fixture(scope="session")
def cluster_bind(cluster):
    if cluster == "deep":
        cluster_bind = (
            "/scratch:/scratch,/opt:/opt,/lnm:/lnm,/bin:/bin,/etc:/etc/,/lib:/lib,/lib64:/lib64"
        )
    elif cluster == "bruteforce":
        # pylint: disable=line-too-long
        cluster_bind = "/scratch:/scratch,/opt:/opt,/lnm:/lnm,/cluster:/cluster,/bin:/bin,/etc:/etc/,/lib:/lib,/lib64:/lib64"
        # pylint: enable=line-too-long
    return cluster_bind


@pytest.fixture(scope="session")
def scheduler_type(cluster):
    """ Switch type of scheduler according to cluster. """
    if cluster == "deep":
        scheduler_type = "pbs"
    elif cluster == "bruteforce":
        scheduler_type = "slurm"
    return scheduler_type


@pytest.fixture(scope="session")
def cluster_queens_testing_folder(cluster_user):
    """ Generic folder on cluster for testing. """
    cluster_queens_testing_folder = pathlib.Path("/home", cluster_user, "queens-testing")
    return cluster_queens_testing_folder


@pytest.fixture(scope="session")
def cluster_path_to_singularity(cluster_queens_testing_folder):
    """ Folder on cluster where to put the singularity file. """
    cluster_path_to_singularity = cluster_queens_testing_folder.joinpath("singularity")
    return cluster_path_to_singularity


@pytest.fixture(scope="session")
def prepare_cluster_testing_environment(
    cluster_user, cluster_address, cluster_queens_testing_folder, cluster_path_to_singularity
):
    """ Create a clean testing environment on the cluster. """
    # remove old folder
    print(f"Delete testing folder from {cluster_address}")
    command_string = f'rm -rfv {cluster_queens_testing_folder}'
    returncode, pid, stdout, stderr = run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_user=cluster_user,
        remote_address=cluster_address,
    )
    print(stdout)
    if returncode:
        raise Exception(stderr)

    # create generic testing folder
    print(f"Create testing folder on {cluster_address}")
    command_string = f'mkdir -v -p {cluster_queens_testing_folder}'
    returncode, pid, stdout, stderr = run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_user=cluster_user,
        remote_address=cluster_address,
    )
    print(stdout)
    if returncode:
        raise Exception(stderr)

    # create folder for singularity
    print(f"Create folder for singularity image on {cluster_address}")
    command_string = f'mkdir -v -p {cluster_path_to_singularity}'
    returncode, pid, stdout, stderr = run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_user=cluster_user,
        remote_address=cluster_address,
    )
    print(stdout)
    if returncode:
        raise Exception(stderr)

    return True


@pytest.fixture(scope="session")
def prepare_singularity(
    connect_to_resource,
    cluster_bind,
    cluster_path_to_singularity,
    prepare_cluster_testing_environment,
):
    """ Build singularity based on the code during test invokation.

    WARNING: needs to be done AFTER prepare_cluster_testing_environment to make sure cluster testing
     folder is clean and existing
    """
    if not prepare_cluster_testing_environment:
        raise RuntimeError("Testing environment on cluster not successfull.")

    remote_flag = True
    singularity_manager = SingularityManager(
        remote_flag=remote_flag,
        connect_to_resource=connect_to_resource,
        cluster_bind=cluster_bind,
        path_to_singularity=str(cluster_path_to_singularity),
    )
    singularity_manager.check_singularity_system_vars()
    singularity_manager.prepare_singularity_files()
    return True


@pytest.fixture(scope="session")
def cluster_testsuite_settings(
    cluster,
    cluster_user,
    cluster_address,
    cluster_bind,
    connect_to_resource,
    cluster_queens_testing_folder,
    cluster_path_to_singularity,
    prepare_singularity,
    scheduler_type,
):
    """ Collection of settings needed for all cluster tests. """
    if not prepare_singularity:
        raise RuntimeError(
            "Preparation of singularity for cluster failed."
            "Make sure to prepare singularity image before using this fixture. "
        )
    cluster_testsuite_settings = dict()
    cluster_testsuite_settings["cluster"] = cluster
    cluster_testsuite_settings["cluster_user"] = cluster_user
    cluster_testsuite_settings["cluster_address"] = cluster_address
    cluster_testsuite_settings["cluster_bind"] = cluster_bind
    cluster_testsuite_settings["connect_to_resource"] = connect_to_resource
    cluster_testsuite_settings["cluster_queens_testing_folder"] = cluster_queens_testing_folder
    cluster_testsuite_settings["cluster_path_to_singularity"] = cluster_path_to_singularity
    cluster_testsuite_settings["scheduler_type"] = scheduler_type

    return cluster_testsuite_settings


@pytest.fixture(scope="session")
def baci_cluster_paths(cluster_user, cluster_address):
    path_to_executable = pathlib.Path("/home", cluster_user, "workspace", "build", "baci-release")
    path_to_postprocessor = pathlib.Path(
        "/home", cluster_user, "workspace", "build", "post_drt_monitor"
    )
    command_string = f'test -f {path_to_executable}'
    returncode, _, stdout, stderr = run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_user=cluster_user,
        remote_address=cluster_address,
    )
    if returncode:
        raise RuntimeError(
            f"Could not find executable on {cluster_address}.\n"
            f"Was looking here: {path_to_executable}"
        )

    command_string = f'test -f {path_to_postprocessor}'
    returncode, _, stdout, stderr = run_subprocess(
        command_string=command_string,
        subprocess_type='remote',
        remote_user=cluster_user,
        remote_address=cluster_address,
    )
    if returncode:
        raise RuntimeError(
            f"Could not find postprocessor on {cluster_address}.\n"
            f"Was looking here: {path_to_postprocessor}"
        )
    baci_cluster_paths = dict(
        path_to_executable=path_to_executable, path_to_postprocessor=path_to_postprocessor
    )
    return baci_cluster_paths