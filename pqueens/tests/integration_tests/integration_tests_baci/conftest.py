import os
import pathlib

import pytest


@pytest.fixture(autouse=True)
def setup_symbolic_links_baci(config_dir, baci_link_paths, baci_source_paths_for_gitlab_runner):
    """
    Set-up of BACI symoblic links

    Args:
        inputdir (str): Path to the JSON input file
        tmpdir (str): Temporary directory in which the pytests are run
        third_party_inputs (str): Path to the BACI input files
        config_dir (str): Path to the config directory of QUEENS containing BACI executables
        set_baci_links_for_gitlab_runner (str): Several paths that are needed to build symbolic
                                                links to executables


    """
    dst_baci_release, dst_post_drt_monitor = baci_link_paths

    src_baci_release, src_post_drt_monitor = baci_source_paths_for_gitlab_runner
    # check if symbolic links are existent
    try:
        # create link to default baci-release location if no link is available
        if not os.path.islink(dst_baci_release):
            if not os.path.isfile(src_baci_release):
                raise FileNotFoundError(
                    f'Failed to create link to default baci-release location.\n'
                    f'No baci-release found under default location:\n'
                    f'\t{src_baci_release}\n'
                )
            else:
                os.symlink(src_baci_release, dst_baci_release)
        # create link to default post_drt_monitor location if no link is available
        if not os.path.islink(dst_post_drt_monitor):
            if not os.path.isfile(src_post_drt_monitor):
                raise FileNotFoundError(
                    f'Failed to create link to default post_drt_monitor location.\n'
                    f'No post_drt_monitor found under default location:\n'
                    f'\t{src_post_drt_monitor}\n'
                )
            else:
                os.symlink(src_post_drt_monitor, dst_post_drt_monitor)
        # check if exitisting link to baci-release works and points to a valid file
        if not pathlib.Path(dst_baci_release).resolve().exists():
            raise FileNotFoundError(
                f'The following link seems to be dead: {dst_baci_release}\n'
                f'It points to (non-existing): {pathlib.Path(dst_baci_release).resolve()}\n'
            )
        # check if exitisting link to post_drt_monitor works and points to a valid file
        if not pathlib.Path(dst_post_drt_monitor).resolve().exists():
            raise FileNotFoundError(
                f'The following link seems to be dead: {dst_post_drt_monitor}\n'
                f'It points to: {pathlib.Path(dst_post_drt_monitor).resolve()}\n'
            )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f'{error}' + 'Please make sure to make the missing executable availabe under the given '
            'path OR\n'
            'make sure the symbolic link under the config directory points to an '
            'existing file! \n'
            'You can create the the necessary symbolic links on Linux via:\n'
            '-------------------------------------------------------------------------\n'
            'ln -s <path/to/baci-release> <QUEENS_BaseDir>/config/baci-release\n'
            'ln -s <path/to/post_drt_monitor> '
            '<QUEENS_BaseDir>/config/post_drt_monitor\n'
            '-------------------------------------------------------------------------\n'
        )
