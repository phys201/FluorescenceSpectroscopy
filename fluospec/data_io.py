#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:19:25 2022

@author: brodi
"""
from typing import Union
import pandas as pd
from pathlib import Path, PurePath
        

# TODO: metadata when we have a better idea what final data will look like

def get_data_file_path(filename: str,
                       data_dir: str = 'fluospec/sim_data'
    ) -> Path:
    """
    Constructs path to data file given filename
    
    Parameters:
    -----------
    filename : str
        string of data filename
 
    Returns
    -------
    Path:
        Path to data
    """
    pwd = Path.cwd()
    data_path = pwd/Path(data_dir)/Path(filename)
    return data_path
    

def load_data(data_file: Union[str, Path]
    ) -> pd.DataFrame:
    """s
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
    if isinstance(data_file, str):
        data_file = Path(data_file)
        
    suffix = data_file.suffix
    
    if suffix == '.pkl':
        return pd.read_pickle(data_file)
    if suffix == '.csv':
        return pd.read_csv(data_file,
                           header=None,
                           names=['w', 'I', 'sigma_I']
                           )
    
    else: 
        raise ValueError(f"Fileype {suffix.strip('.')} not supported, only pkl and csv.")
    


                         
    