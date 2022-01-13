"""Test Sobol indices estimation with Gaussian process surrogate."""
import os
import pickle

import numpy as np

from pqueens.main import main


def test_sobol_indices_ishigami_gp(inputdir, tmpdir):
    """Test Sobol indices estimation with Gaussian process surrogate."""
    arguments = [
        '--input=' + os.path.join(inputdir, 'sobol_indices_ishigami_gp.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    expected_result_s1 = np.array([0.28977645, 0.43374979, -0.04328034])
    expected_result_s1_conf = np.array([0.20741671, 0.17213406, 0.10860589])

    np.testing.assert_allclose(results['sensitivity_indices']['S1'], expected_result_s1, atol=1e-05)
    np.testing.assert_allclose(
        results['sensitivity_indices']['S1_conf'], expected_result_s1_conf, atol=1e-05
    )