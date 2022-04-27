
import unittest
from unittest import TestCase
from pathlib import PurePath

from fluospec.data_io import load_data, get_data_file_path

from pandas import DataFrame


df_cols = ['w', 'I', 'sigma_I']




class TestIO(TestCase):
    sim_data_path = get_data_file_path('fluospec_sim_data.pkl')
    def test_get_fi_path_returns_Path(self):
        self.assertTrue(isinstance(self.sim_data_path, PurePath))
        
    def test_data_is_df(self):
        data_df = load_data(self.sim_data_path)
        self.assertTrue(isinstance(data_df, DataFrame))

    def test_df_columns(self):
        data_df = load_data(self.sim_data_path)
        self.assertCountEqual(data_df.keys(), df_cols)

    def test_df_dim(self):
        data_df = load_data(self.sim_data_path)
        self.assertEqual(data_df.shape[-1], 3)
        
    def test_invalid_extension(self):
         with self.assertRaises(ValueError):
             bad_path = get_data_file_path('fluospec_sim_data.bad')
             load_data(bad_path)
             

        

if __name__ == '__main__':
    unittest.main()
