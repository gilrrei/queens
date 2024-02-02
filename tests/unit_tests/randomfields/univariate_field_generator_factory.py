"""Created on April 19th 2017.

@author: jbi
"""

import unittest

import numpy as np
from scipy.stats import beta, norm

from queens.randomfields.univariate_random_field_factory import create_univariate_random_field


class TestRandomFieldGeneratorConstructionFactory(unittest.TestCase):
    """TODO_doc."""

    def setUp(self):
        """TODO_doc."""
        # setup some necessary variables to setup random field generators
        self.dimension = 1
        self.corrstruct = 'squared_exp'
        self.corr_length = 25
        self.energy_frac = 0.95
        self.field_bbox = np.array([-100, 100])
        self.marginal_pdf = norm(0, 1)
        self.num_terms_per_dim = 120
        self.total_terms = 120
        self.loc = [0, 10, 25, 100]
        self.seed = 42

    def test_construction_wrong_distribution(self):
        """TODO_doc."""
        with self.assertRaises(RuntimeError):
            mystuff = create_univariate_random_field(
                beta(2, 4),
                self.corrstruct,
                self.dimension,
                self.corr_length,
                self.energy_frac,
                self.field_bbox,
                self.num_terms_per_dim,
                self.total_terms,
            )
            mystuff.gen_sample_gauss_field(10, np.array((4, 4)))

    def test_construction_wrong_covariance(self):
        """TODO_doc."""
        with self.assertRaises(RuntimeError):
            mystuff = create_univariate_random_field(
                self.marginal_pdf,
                'weird_stuff',
                self.dimension,
                self.corr_length,
                self.energy_frac,
                self.field_bbox,
                self.num_terms_per_dim,
                self.total_terms,
            )
            mystuff.gen_sample_gauss_field(10, np.array((4, 4)))

    def test_construction_wrong_dimension(self):
        """TODO_doc."""
        with self.assertRaises(ValueError):
            create_univariate_random_field(
                self.marginal_pdf,
                self.corrstruct,
                4,  # should throw error
                self.corr_length,
                self.energy_frac,
                self.field_bbox,
                self.num_terms_per_dim,
                self.total_terms,
            )
        with self.assertRaises(ValueError):
            create_univariate_random_field(
                self.marginal_pdf,
                'exp',
                4,  # should throw error
                self.corr_length,
                self.energy_frac,
                self.field_bbox,
                self.num_terms_per_dim,
                self.total_terms,
            )


class TestRandomFieldGeneratorFourierConstruction(unittest.TestCase):
    """TODO_doc."""

    def setUp(self):
        """TODO_doc."""
        # setup some necessary variables to setup random field generators
        self.dimension = 1
        self.corrstruct = 'squared_exp'
        self.corr_length = 25
        self.energy_frac = 0.95
        self.field_bbox = np.array([-100, 100])
        self.marginal_pdf = norm(0, 1)
        self.num_terms_per_dim = 120
        self.total_terms = 120
        self.loc = [0, 10, 25, 100]
        self.seed = 42

    # raise RuntimeError('field bounding box must be size {} and not {}'.format(
    # self.spatial_dim*2,san_check_bbox[0]))
    def test_construction_boundingbox_dimension(self):
        """TODO_doc."""
        with self.assertRaises(ValueError):
            create_univariate_random_field(
                self.marginal_pdf,
                self.corrstruct,
                self.dimension,
                self.corr_length,
                self.energy_frac,
                np.array([100, 100, 100]),
                self.num_terms_per_dim,
                self.total_terms,
            )

    # raise ValueError('energy fraction must be between 0 and 1.')
    def test_construction_wrong_engergy_frac(self):
        """TODO_doc."""
        with self.assertRaises(ValueError):
            create_univariate_random_field(
                self.marginal_pdf,
                self.corrstruct,
                self.dimension,
                self.corr_length,
                1.1,
                self.field_bbox,
                self.num_terms_per_dim,
                self.total_terms,
            )

            mystuff = create_univariate_random_field(
                self.marginal_pdf,
                self.corrstruct,
                self.dimension,
                self.corr_length,
                -1.1,
                self.field_bbox,
                self.num_terms_per_dim,
                self.total_terms,
            )
            mystuff.gen_sample_gauss_field(10, np.array((4, 4)))

    # raise ValueError('correlation length must smaller than
    def test_construction_correlation_length_to_long(self):
        """TODO_doc."""
        with self.assertRaises(ValueError):
            create_univariate_random_field(
                self.marginal_pdf,
                self.corrstruct,
                self.dimension,
                71,
                self.energy_frac,
                self.field_bbox,
                self.num_terms_per_dim,
                self.total_terms,
            )

    # raise RuntimeError('Error: correlation length must be positive')
    def test_construction_correlation_length_negative(self):
        """TODO_doc."""
        with self.assertRaises(ValueError):
            create_univariate_random_field(
                self.marginal_pdf,
                self.corrstruct,
                self.dimension,
                -10,
                self.energy_frac,
                self.field_bbox,
                self.num_terms_per_dim,
                self.total_terms,
            )


class TestRandomFieldGeneratorKLEConstruction(unittest.TestCase):
    """TODO_doc."""

    def setUp(self):
        """TODO_doc."""
        # setup some necessary variables to setup random field generators
        self.dimension = 1
        self.corrstruct = 'exp'
        self.corr_length = 25
        self.energy_frac = 0.95
        self.field_bbox = np.array([-100, 100])
        self.marginal_pdf = norm(0, 1)
        self.num_terms_per_dim = 120
        self.total_terms = 120
        self.loc = [0, 10, 25, 100]
        self.seed = 42

    # raise RuntimeError('field bounding box must be size {} and not {}'.format(
    # self.spatial_dim*2,san_check_bbox[0]))
    def test_construction_boundingbox_dimension(self):
        """TODO_doc."""
        with self.assertRaises(ValueError):
            create_univariate_random_field(
                self.marginal_pdf,
                self.corrstruct,
                self.dimension,
                self.corr_length,
                self.energy_frac,
                np.array([-100, 100, 100]),
                self.num_terms_per_dim,
                self.total_terms,
            )

    # raise RuntimeError Number of terms in KLE expansion is too large. '
    def test_num_expansion_terms(self):
        """TODO_doc."""
        with self.assertRaises(ValueError):
            create_univariate_random_field(
                self.marginal_pdf,
                self.corrstruct,
                2,
                self.corr_length,
                self.energy_frac,
                np.array([-100, 100, -100, 100]),
                10,
                120,
            )

    # raise ValueError('energy fraction must be between 0 and 1.')
    def test_construction_wrong_engergy_frac(self):
        """TODO_doc."""
        with self.assertRaises(ValueError):
            create_univariate_random_field(
                self.marginal_pdf,
                self.corrstruct,
                self.dimension,
                self.corr_length,
                1.1,
                self.field_bbox,
                self.num_terms_per_dim,
                self.total_terms,
            )

            mystuff = create_univariate_random_field(
                self.marginal_pdf,
                self.corrstruct,
                self.dimension,
                self.corr_length,
                -1.1,
                self.field_bbox,
                self.num_terms_per_dim,
                self.total_terms,
            )
            mystuff.gen_sample_gauss_field(10, np.array((4, 4)))

    # raise RuntimeError('Error: correlation length must be positive')
    def test_construction_correlation_length_negative(self):
        """TODO_doc."""
        with self.assertRaises(ValueError):
            create_univariate_random_field(
                self.marginal_pdf,
                self.corrstruct,
                self.dimension,
                -10,
                self.energy_frac,
                self.field_bbox,
                self.num_terms_per_dim,
                self.total_terms,
            )


if __name__ == '__main__':
    unittest.main()