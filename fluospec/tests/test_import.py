import unittest
from unittest import TestCase


import fluospec.model
from fluospec.data_io import load_data

class TestModel(TestCase):
    def test_is_Prediction(self):
        default_prediction = fluospec.model.Prediction.init_with_defaults()
        self.assertTrue(isinstance(default_prediction, fluospec.model.Prediction))
        
    
        
if __name__ == '__main__':
    unittest.main()