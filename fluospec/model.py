#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:19:25 2022

@author: brodi
"""
from typing import Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import pymc3 as pm
from dataclasses import dataclass
from pymc3.model import FreeRV

@dataclass
class Prediction():
    '''
    Helper class that contains the generators for the model 
    '''
    A: Union[float, FreeRV]
    w0: Union[float, FreeRV]
    gamma: Union[float, FreeRV]
    intensity_ratio: Union[float, FreeRV]
    m: Union[float, FreeRV]
    b: Union[float, FreeRV]
    
    @classmethod
    def init_with_defaults(cls):
        theta = {'A': 2,
                 'w0': 20,
                 'gamma': 5,
                 'intensity_ratio': .5,
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
        w: np.ndarray
            Array of frequencies over which to take Lorentzian
 
        Returns
        -------
        ndarray:
            array of the Lorentzian of the parameters  
        """
        lorentzian = self.A*(self.gamma**2/4)/((w - self.w0)**2 + \
                    (self.gamma**2/4)*(1+self.intensity_ratio))
        return lorentzian
    
    def prediction(self,
                   w: np.ndarray
        ) -> np.ndarray:
        """
        Calculates the model prediction line, which includes a power-aware
        Lorentzian and a background lineshape from spectrometer drift
        
        Parameters:
        -----------
        w: np.ndarray
            Array of frequencies over which to take the model
 
        Returns
        -------
        ndarray:
            array of model prediction points 
        """
        lorentzian = self.lorentzian(w)
        line = self.m*w + self.b
        model_prediction = line + lorentzian
        return model_prediction
    

class SimulateFluoSpec(Prediction): 
    '''
    Subclass of prediction, as data generated from generative model
    Thus, instantiate with known model parameters
    '''
    def generate_sim_data(self, 
                          data_range: Tuple[int, int] = (0, 40),
                          data_unc: float = .2
        ) -> pd.DataFrame:
        """
        Generates the simulated data from the model, taking model lineshape
        and adding Gaussian noise
        
        Parameters:
        -----------
        data_range: tuple
            (lower, upper) tuple of ranges to generate simulated frequencies
            
        data_unc:
            Data uncertainty, assuming Gaussian noise and a single uncertainty
            for all data points
            
 
        Returns
        -------
        ndarray:
            dataframe of simulated data
        """
        w_sim = np.linspace(*data_range, 250)

        prediction = self.prediction(w_sim)
        
        noise = np.random.normal(0, data_unc, len(w_sim))
        
        sim_data = (prediction + noise)
        
        return pd.DataFrame({'w': w_sim,
                             'I': sim_data,
                             'sigma_I': data_unc
                             })
    
    def save_sim_data(self,
                      data_range: Tuple[int, int] = (0, 40),
                      data_unc: float = .2
        ) -> None:
        """
        Saves simulated data using pickle to pwd
        
        Parameters:
        -----------
        data_range: tuple
            (lower, upper) tuple of ranges to generate simulated frequencies
            
        data_unc:
            Data uncertainty, assuming Gaussian noise and a single uncertainty
            for all data points
            
 
        Returns
        -------
        """
        pwd = Path.cwd()
        save_path = pwd/Path("fluospec_sim_data.pkl")
        sim_data = self.generate_sim_data(data_range, data_unc)
        sim_data.to_pickle(save_path)
  
    
@dataclass    
class FluoSpecModel():
    '''
    Class for fluospec model. Instantiate with parameters on normal prior, as
    tuple with (mean, std)
    
    Run `FluoSpecModel?` to see order. 
    '''
    A_prior_params: Tuple
    w0_prior_params: Tuple
    gamma_prior_params: Tuple
    intensity_ratio_prior_params: Tuple
    m_prior_params: Tuple
    b_prior_params: Tuple
    
        
    def model(self,
              spec_data_df: pd.DataFrame,
        ) -> pm.Model:
        """
        Builds generative model for fluorescence spectroscopy
        
        Parameters:
        -----------
        spec_data_df: pd.DataFrame
            dataframe of data to build model for
            
 
        Returns
        -------
        pm.Model:
            pymc3 model
        """
        spectroscopy_model = pm.Model()
    
        w_data = spec_data_df.w.to_numpy()
        I_data = spec_data_df.I
        sigma_I_data = spec_data_df.sigma_I
        
        with spectroscopy_model:
            # define priors
            A = pm.Normal('A', *self.A_prior_params)
            w0 = pm.Normal('w0', *self.w0_prior_params)
            gamma = pm.Normal('gamma', *self.gamma_prior_params)
            intensity_ratio = pm.Normal('intensity_ratio',
                                        *self.intensity_ratio_prior_params)
            
            m = pm.Normal('m', *self.m_prior_params)
            b = pm.Normal('b', *self.b_prior_params)
            
            # convenient prior vector
            theta = (A, w0, gamma, intensity_ratio, m, b)
                
            # predicted intensity
            I_pred = pm.Deterministic('prediction',
                                      Prediction(*theta).prediction(w_data))
         
            # measurements have normal noise
            measurements = pm.Normal('I_model',
                                     mu=I_pred,
                                     sigma=sigma_I_data,
                                     observed=I_data)
            
        return spectroscopy_model

    