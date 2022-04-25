
import unittest
from unittest import TestCase

from fluospec.data_io import load_data

from pathlib import Path
from pandas import DataFrame


df_cols = ['w', 'I', 'sigma_I']


class TestIO(TestCase):
    def test_data_is_df(self):
        data_df = load_data(Path('../sim_data/fluospec_sim_data.csv'))
        self.assertTrue(isinstance(data_df, DataFrame))
    
    def test_df_columns(self):
        data_df = load_data(Path('../sim_data/fluospec_sim_data.csv'))
        self.assertCountEqual(data_df.keys(), df_cols)
        
    def test_df_dim(self):
        data_df = load_data(Path('../sim_data/fluospec_sim_data.csv'))
        self.assertEqual(data_df.shape[-1], 3)

if __name__ == '__main__':
    unittest.main()