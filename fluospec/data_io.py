#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:19:25 2022

@author: brodi
"""
from typing import Union
import pandas as pd
from pathlib import Path
        

# TODO: metadata when we have a better idea what final data will look like

def load_data(data_file: Union[str, Path],
    ) -> pd.DataFrame:
    """
    Loads experimental or simulated data in csv format give filepath name
    
    Parameters:
    -----------
    data_file : Union[str, Path]
        string or Path to csv of data
 
    Returns
    -------
    pd.DataFrame:
        DataFrame containing columns of data (w, I, sigma_I)
    """
    return pd.read_csv(data_file,
                       header=None,
                       names=['w', 'I', 'sigma_I']
                       )