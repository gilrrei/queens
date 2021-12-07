import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from pqueens.main import main
from pqueens.tests.integration_tests.example_simulator_functions.park91a_hifi_coords import (
    park91a_hifi_coords,
)
from pqueens.utils import injector


@pytest.mark.benchmark
def test_smc_park_hf(inputdir, tmpdir, design_and_write_experimental_data_to_csv):
    """Integration test for bayesian multi-fidelity inverse analysis (bmfia)
    using the park91 function."""

    # generate json input file from template
    template = os.path.join(inputdir, 'bmfia_smc_park.json')
    experimental_data_path = tmpdir
    dir_dict = {'experimental_data_path': experimental_data_path}
    input_file = os.path.join(tmpdir, 'smc_mf_park_realization.json')
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    arguments = [
        '--input=' + input_file,
        '--output=' + str(tmpdir),
    ]

    # actual main call of smc
    main(arguments)

    # get the results of the QUEENS run
    result_file = os.path.join(tmpdir, 'smc_park_mf.pickle')
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    samples = results['raw_output_data']['particles'].squeeze()
    weights = results['raw_output_data']['weights'].squeeze()

    # quick and dirty plotting
    sns.set_theme(style='whitegrid')
    f, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=samples[:, 0], y=samples[:, 1], s=5, color='.15')
    sns.kdeplot(
        x=samples[:, 0],
        y=samples[:, 1],
        weights=weights,
        thresh=0.2,
        levels=4,
        color='k',
        linewidths=1,
    )
    ax.plot(0.5, 0.2, 'x', color='red', label='ground truth')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    plt.savefig('/home/nitzler/park_mf_smc.jpg', dpi=300)

    # Actual test
    f2, ax2 = plt.subplots(figsize=(6, 6))
    noise_var_lst = results['raw_output_data']['noise_var_lst']
    ax2.plot(noise_var_lst, '.')
    ax2.set_xlabel('iter')
    ax2.set_ylabel(r'$\sigma^2_{\mathrm{obs,MAP}}$')
    plt.savefig('/home/nitzler/noise_var_lst.jpg', dpi=300)


# assert np.abs(results['variational_distr']['mu'][0] - 0.5) < 0.01
# assert np.abs(results['variational_distr']['mu'][1] - 0.2) < 0.03
# assert results['variational_distr']['sigma'][0] < 0.1
# assert results['variational_distr']['sigma'][1] < 0.3
# assert results['variational_distr']['noise_std'] < 0.05
# assert results['iterations'] < 10000


@pytest.fixture()
def design_and_write_experimental_data_to_csv(tmpdir):
    # Fix random seed
    np.random.seed(seed=1)

    # create target inputs
    x1 = 0.5
    x2 = 0.2

    # use x3 and x4 as coordinates and create coordinate grid (same as in park91a_hifi_coords)
    xx3 = np.linspace(0.0, 1.0, 4)
    xx4 = np.linspace(0.0, 1.0, 4)
    x3_vec, x4_vec = np.meshgrid(xx3, xx4)
    x3_vec = x3_vec.flatten()
    x4_vec = x4_vec.flatten()

    # generate clean function output for fake test data
    y_vec = []
    for x3, x4 in zip(x3_vec, x4_vec):
        y_vec.append(park91a_hifi_coords(x1, x2, x3, x4))
    y_vec = np.array(y_vec)

    # add artificial noise to fake measurements
    sigma_n = 0.1
    noise_vec = np.random.normal(loc=0, scale=sigma_n, size=(y_vec.size,))
    y_fake = y_vec + noise_vec

    # write fake data to csv
    data_dict = {
        'x3': x3_vec,
        'x4': x4_vec,
        'y_obs': y_fake,
    }
    experimental_data_path = os.path.join(tmpdir, 'experimental_data.csv')
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)
