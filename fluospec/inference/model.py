#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:19:25 2022

@author: brodi
"""
from typing import Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import pymc3 as pm
from dataclasses import dataclass

@dataclass
class Prediction():
    A: Union[float, pm.Modeli
    w0: float
    gamma: float
    m: float
    b: float
    
    @classmethod
    def init_with_defaults(cls):
        theta = {'A': 20,
                 'w0': 1,
                 'gamma': 5,
                 'm': .05,
                 'b': 2}
        return cls(**theta)
    
    def lorentzian(self,
                   w: np.ndarray,
         ) -> np.ndarray: 
        """
        Calculates the Lorentzian line shape 
        
        Parameters:
        -----------
        A : float
            Amplitude of line 
        w : ndarray
            Frequency
        w0 : ndarray
            Central frequency
        gamma : ndarray
            The decay rate of the transition. 
            Also the full width at half maximum      
        Returns
        -------
        ndarray:
            array of the Lorentzian of the parameters  
        """
        lorentzian = self.A*(self.gamma**2/4)/((w - self.w0)**2 + (self.gamma**2/4))
        return lorentzian
    
    def prediction(self,
                   w: np.ndarray
        ) -> np.ndarray:
        # w0, A, gamma, m, b = theta
        lorentzian = self.lorentzian(w)
        line = self.m*w + self.b
        model_prediciton = line + lorentzian
        return model_prediciton
    

class SimulateFluoSpec(Prediction): 
    def generate_sim_data(self, 
                          data_range: Tuple[int, int] = (0, 40),
                          data_unc: float = .2
        ) -> pd.DataFrame:
        w_sim = np.linspace(*data_range, 250)

        prediction = self.prediction(w_sim)
        
        noise = np.random.normal(0, data_unc, len(w_sim))
        
        sim_data = (prediction + noise)
        
        return pd.DataFrame({'w': w_sim,
                             'I': sim_data,
                             'sigma_I': data_unc
                             })
    
    def save_sim_data(self,
                      save_path: Path,
                      data_range: Tuple[int, int] = (0, 40),
                      data_unc: float = .2
        ) -> None:
        sim_data = self.generate_sim_data(data_range, data_unc)
        sim_data.to_csv(save_path)
  
        
class FluoSpecModel():
    def __init__(self):
        pass
        
    def model(self,
              spec_data_df: pd.DataFrame,
        ) -> pm.Model:
        spectroscopy_model = pm.Model()
    
        w_data = spec_data_df.w.as_array()
        I_data = spec_data_df.I
        sigma_I_data = spec_data_df.sigma_I
        
        with spectroscopy_model:
            A = pm.Uniform('A', 0, 5)
            w0 = pm.Uniform('w0', 10, 30)
            gamma = pm.Uniform('gamma', 0, 10)
            
            m = pm.Uniform('m', 0, 0.1)
            b = pm.Uniform('b', 0, 3)
            
            theta = (A, w0, gamma, m, b)
                
            I_pred = pm.Deterministic('prediction',
                                      self.prediciton(w_data, theta))
         
            measurements = pm.Normal('I_model',
                                     mu=I_pred,
                                     sigma=sigma_I_data,
                                     observed=I_data)
            
        return spectroscopy_model

    