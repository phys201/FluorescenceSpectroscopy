import unittest
from unittest import TestCase

from pathlib import Path

from fluospec.model import Prediction, FluoSpecModel, SimulateFluoSpec, SpectralLines
from fluospec.data_io import load_data, get_data_file_path
from pymc3 import Model
from os.path import exists

theta = {'spectral_lines': [
                           SpectralLines(2,
                                         30,
                                         5,
                                         .5
                                        ),
                            SpectralLines(1,
                                          10,
                                          4,
                                          .5
                                         )
                           ],
         'm': .05,
         'b': 2,
         }

class TestSpectralLines(TestCase):
    spec_line = theta['spectral_lines'][0]
    def test_SpectralLines(self):
        keys = list(vars(self.spec_line).keys())
        self.assertEqual(keys,
                         ['A', 'w0', 'gamma', 'intensity_ratio'])

class TestModel(TestCase):
    sim_data_path = get_data_file_path('fluospec_sim_data.pkl')
    def test_init_with_defaults(self):
        pred = Prediction.init_with_defaults()
        self.assertEqual(pred, Prediction(**theta))

    def test_lorentzian(self):
        pred = Prediction.init_with_defaults()
        spec_line = pred.spectral_lines[0]
        w_res = spec_line.w0
        spec_line.intensity_ratio = 0
        loren = spec_line.lorentzian(w_res)
        self.assertEqual(loren, spec_line.A)

    def test_prediction_line(self):
        w = 1
        pred = Prediction.init_with_defaults()
        spec_line_0 = pred.spectral_lines[0]
        spec_line_1 = pred.spectral_lines[1]
        spec_line_0.A = 0
        spec_line_1.A = 0
        line = pred.m*w + pred.b
        self.assertEqual(pred.prediction(w), line)

    def test_prediction_lorentzian(self):
        pred = Prediction.init_with_defaults()
        spec_line_0 = pred.spectral_lines[0]
        spec_line_1 = pred.spectral_lines[1]
        pred.m = 0
        pred.b = 0
        self.assertEqual(pred.prediction(1),
                         spec_line_0.lorentzian(1) + spec_line_1.lorentzian(1))

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
                            'line_prior_params': [
                                                ((2, 1), (30, 2), (5, 1), (.5, .1)),
                                                ((1, 1), (10, 2), (4, .5), (.5, .1))
                                                 ],
                            'm_prior_params': (.0, .01),
                            'b_prior_params': (2, .04),
                            }
        data_df = load_data(self.sim_data_path)
        fluospec_model_instance = FluoSpecModel(**model_sim_params)
        generative_model = fluospec_model_instance.model(data_df)
        self.assertTrue(isinstance(generative_model, Model))

if __name__ == '__main__':
    unittest.main()
