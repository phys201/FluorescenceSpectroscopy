import unittest
from unittest import TestCase

from pathlib import Path

from fluospec.model import Prediction
from fluospec.model import FluoSpecModel
from  fluospec.model import SimulateFluoSpec
from fluospec.data_io import load_data, get_data_file_path
from pymc3 import Model
from os.path import exists

theta = {'A': 2,
         'w0': 20,
         'gamma': 5,
         'intensity_ratio': .5,
         'm': .05,
         'b': 2}

class TestModel(TestCase):
    sim_data_path = get_data_file_path('fluospec_sim_data.pkl')
    def test_init_with_defaults(self):
        pred = Prediction.init_with_defaults()
        self.assertEqual(pred, Prediction(**theta))

    def test_lorentzian(self):
        pred = Prediction.init_with_defaults()
        w_res = pred.w0
        pred.intensity_ratio = 0
        loren = pred.lorentzian(w_res)
        self.assertEqual(loren, pred.A)

    def test_prediction_line(self):
        w = 1
        pred = Prediction.init_with_defaults()
        pred.A = 0
        line = pred.m*w + pred.b
        self.assertEqual(pred.prediction(w), line)

    def test_prediction_lorentzian(self):
        pred = Prediction.init_with_defaults()
        pred.m = 0
        pred.b = 0
        self.assertEqual(pred.prediction(1), pred.lorentzian(1))

    def test_generate_sim_data(self):
        sim = SimulateFluoSpec.init_with_defaults()
        sim_data = sim.generate_sim_data()
        column_names = ['w', 'I', 'sigma_I']
        self.assertEqual(list(sim_data.columns),column_names)

    def test_save_sim_data_fi_exists(self):
        try:
            sim = SimulateFluoSpec.init_with_defaults()
            sim.save_sim_data()
            pwd = Path.cwd()
            save_path = pwd/Path("fluospec_sim_data.pkl")
            self.assertTrue(exists(save_path))
        finally:
            save_path.unlink()


    def test_FluoSpecModel_returns_Model(self):
        model_sim_params = {
                    'A_prior_params': (2, 1),
                    'w0_prior_params': (20, 4),
                    'gamma_prior_params': (5, 1),
                    'intensity_ratio_prior_params': (.5, .1),
                    'm_prior_params': (.05, .01),
                    'b_prior_params': (2, .04),
                    }
        data_df = load_data(self.sim_data_path)
        fluospec_model_instance = FluoSpecModel(**model_sim_params)
        generative_model = fluospec_model_instance.model(data_df)
        self.assertTrue(isinstance(generative_model, Model))

if __name__ == '__main__':
    unittest.main()
