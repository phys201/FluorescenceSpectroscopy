#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:19:25 2022

@author: brodi
"""
from typing import Union
import pandas as pd
from pathlib import Path
        

def load_data(data_file: Union[str, Path]
    ) -> pd.DataFrame:
    return pd.read_csv(data_file)
