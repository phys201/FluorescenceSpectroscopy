#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:19:25 2022

@author: brodi
"""
from typing import Tuple, Union, List
from pathlib import Path
import numpy as np
import pandas as pd
import pymc3 as pm
from dataclasses import dataclass
from pymc3.model import FreeRV
from theano.tensor.var import TensorVariable



@dataclass
class SpectralLines():
    '''
    Class representing a spectral line
    
    Currently implements a Lorentzian line given
        
    amplitude, line center, width, and an intensity ratio
    '''
    A: Union[float, FreeRV]
    w0: Union[float, FreeRV]
    gamma: Union[float, FreeRV]
    intensity_ratio: Union[float, FreeRV]

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
    

@dataclass
class Prediction():
    '''
    Helper class that contains the generators for the model 
    '''
    spectral_lines: Union[List[float], List[SpectralLines]]
    m: Union[float, FreeRV]
    b: Union[float, FreeRV]
    
    def __post_init__(self):
        self.spectral_lines = [
                                SpectralLines(*line) if not isinstance(line, SpectralLines)
                                else line
                                for line in self.spectral_lines
                                ]
    
    
    @classmethod
    def init_with_defaults(cls):
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
        
        return cls(**theta)
    
    
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
        
        lorentzians = [line.lorentzian(w) for line in self.spectral_lines]
        
        lorentzian_arr = np.vstack(lorentzians)
        
        lorentzian_vals = np.sum(lorentzian_arr, axis=0)
        
        if isinstance(lorentzian_vals[0], TensorVariable):
            lorentzian = lorentzian_vals[0]
        else:
            lorentzian = lorentzian_vals
            
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
    line_prior_params: List[Tuple]
    m_prior_params: Tuple
    b_prior_params: Tuple
    scale_prior_params: Tuple = (1)
    
    
    def model(self,
              spec_data_df: pd.DataFrame,
              likelihood: str = 'normal'
        ) -> pm.Model:
        """
        Builds generative model for fluorescence spectroscopy
        
        Parameters:
        -----------
        spec_data_df: pd.DataFrame
            dataframe of data to build model for
            
        likelihood: str
            str indicating the form of the likelihood
            
            'normal': default, good if your noise is well-characterized
            'cauchy': helpful on data that has fine structure
 
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
            line_priors = []
            for i, line_prior in enumerate(self.line_prior_params):
                A_params, w0_params, gamma_params, intensity_params = line_prior
                specline = SpectralLines(
                                              pm.Gamma(f'A_{i}',
                                                       mu=A_params[0],
                                                       sigma=A_params[1]
                                                       ),
                                              
                                               pm.Gamma(f'w0_{i}',
                                                        mu=w0_params[0],
                                                        sigma=w0_params[1]
                                                        ),
                                               pm.Gamma(f'gamma_{i}',
                                                        mu=gamma_params[0],
                                                        sigma=gamma_params[1]
                                                        ),
                                                pm.Gamma(f'intensity_ratio_{i}',
                                                         mu=intensity_params[0],
                                                         sigma=intensity_params[1]
                                                         ),
                                               )
                line_priors.append(specline)
                

            
            m = pm.Normal('m', *self.m_prior_params)
            b = pm.Normal('b', *self.b_prior_params)
            
            # convenient prior vector
            theta = (line_priors, m, b)
                
            # predicted intensity
            I_pred = pm.Deterministic('prediction',
                                      Prediction(*theta).prediction(w_data))
            
            if likelihood == 'normal':
                print('here')
                measurements = pm.Normal('I_model',
                                          mu=I_pred,
                                          sigma=sigma_I_data,
                                          observed=I_data)              
               
            elif likelihood == 'cauchy':
                # measurements modeled as Cauchian noise
                # accounts for broadening of Gaussian instrument noise
                # with undesired background fine structure
                
                scale = pm.TruncatedNormal('scale',
                                           mu=1,
                                           sigma=sigma_I_data.iloc[-1]/10,
                                           lower=sigma_I_data.iloc[-1])
                
                measurements = pm.Cauchy('I_model',
                                         alpha=I_pred,
                                         beta=scale,
                                         observed=I_data)

            else:
                raise ValueError(f'likelihood = {likelihood} is not valid.')
    
        return spectroscopy_model

    # TODO: think about extending above to allow multiple models,
    #       then create helper for model_comparioson
    def model_comparison():
        raise NotImplementedError
    