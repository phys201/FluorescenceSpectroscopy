import unittest
from unittest import TestCase

from fluospec.model import FluoSpecModel
from fluospec.data_io import load_data, get_data_file_path
from pymc3 import Model

model_sim_params = {
                    'A_prior_params': (2, 1),
                    'w0_prior_params': (20, 4),
                    'gamma_prior_params': (5, 1),
                    'intensity_ratio_prior_params': (.5, .1),
                    'm_prior_params': (.05, .01),
                    'b_prior_params': (2, .04),
                    }

sim_data_path = get_data_file_path('fluospec_sim_data.pkl')

class TestModel(TestCase):
    def test_FluoSpecModel_returns_Model(self):
        data_df = load_data(sim_data_path)
        fluospec_model_instance = FluoSpecModel(**model_sim_params)
        generative_model = fluospec_model_instance.model(data_df)
        self.assertTrue(isinstance(generative_model, Model))


if __name__ == '__main__':
    unittest.main()
