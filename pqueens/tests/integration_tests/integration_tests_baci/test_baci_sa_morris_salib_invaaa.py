"""
Test suite for integration tests for the Morris-Salib Iterator (Elementary Effects) for local
simulations with BACI using the INVAAA minimal model.
"""
import os
from pathlib import Path
import numpy as np
import pickle
from pqueens.main import main
from pqueens.utils import injector


def test_baci_morris_salib(inputdir, tmpdir, third_party_inputs, config_dir):
    """Test a morris-salib run with a small BACI simulation model"""
    template = os.path.join(inputdir, "morris_baci_local_invaaa_template.json")
    input_file = os.path.join(tmpdir, "morris_baci_local_invaaa.json")
    third_party_input_file = os.path.join(third_party_inputs, "baci_input_files", "invaaa_ee.dat")

    baci_release = os.path.join(config_dir, "baci-release")
    post_drt_monitor = os.path.join(config_dir, "post_drt_monitor")

    # check if symbolic links are existent
    if (not os.path.islink(baci_release)) or (not os.path.islink(post_drt_monitor)):
        # set default baci location for testing machine
        base_path = Path('$HOME/workspace/baci_release')
        baci_release = os.path.join(base_path, 'baci-release')
        post_drt_monitor = os.path.join(base_path, 'post_drt_monitor')

        # if neither default location works nor symbolic links are existent throw error
        if (not os.path.isfile(baci_release)) or (not os.path.isfile(post_drt_monitor)):
            raise FileNotFoundError('No working baci-release or post_drt_monitor could be found! '
                                    'Make sure an appropriate symbolic link is made available '
                                    'under the config directory! Abort...')

    dir_dict = {
        'experiment_dir': str(tmpdir),
        'baci_input': third_party_input_file,
        'baci-release': baci_release,
        'post_drt_monitor': post_drt_monitor,
    }

    injector.inject(dir_dict, template, input_file)
    arguments = ['--input=' + input_file, '--output=' + str(tmpdir)]
    main(arguments)

    result_file = os.path.join(tmpdir, 'ee_invaaa_local.pickle')
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu"],
        np.array([-0.294384, -1.255711, -0.324267,  0.963429]),
        rtol=1.0e-3,
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star"],
        np.array([0.294384, 1.255711, 0.324267, 0.963429]),
        rtol=1.0e-3,
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["sigma"],
        np.array([0.04812 , 0.125748, 0.039385, 0.14809]),
        rtol=1.0e-3,
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star_conf"],
        np.array([0.024414, 0.069575, 0.025782, 0.087494]),
        rtol=1.0e-3,
    )
