#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:19:25 2022

@author: brodi
"""
from typing import Tuple, Union, List, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import pymc3 as pm
from dataclasses import dataclass, field
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
    def init_with_defaults(cls,
                           n_lines: int = 2
        ):
        """
        Class method to to instantiate Prediction with defaults. 

        Parameters
        ----------
        n_lines : int, optional
            Number of spectral lines, 1-2. The default is 2.

        Returns
        -------
        Prediction
            returns a Prediction object with the specified number of spectral
            lines.

        """
        spec_lines_list = [
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
                        ]
        spectral_lines_to_use = spec_lines_list[:n_lines]
        
        theta = {'spectral_lines': spectral_lines_to_use,
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
                          data_unc: float = .2,
                          mvnormal: bool = False
        ) -> pd.DataFrame:
        """
        Generates the simulated data from the model, taking model lineshape
        and adding Gaussian noise
        
        Parameters:
        -----------
        data_range: tuple
            (lower, upper) tuple of ranges to generate simulated frequencies
            
        data_unc: float
            Data uncertainty, assuming Gaussian noise and a single uncertainty
            for all data points
        
        mvnormal: bool
            If True, simulate data from multivariate normal noise, with
            uncertainty on the cov matrix diagonal, and nearest-neighbor
            correlation
        
            
        Returns
        -------
        ndarray:
            dataframe of simulated data
        """
        w_sim = np.linspace(*data_range, 250)

        prediction = self.prediction(w_sim)
        
        if mvnormal:
            n = len(prediction)
            cov = np.sum(
                           [
                             np.diag(np.ones(n-1)*data_unc/10, -1),
                             np.diag(np.ones(n)*data_unc, 0),
                             np.diag(np.ones(n-1)*data_unc/10, 1),
                           ],
                           axis=0
                         )
            
            sim_data = np.random.multivariate_normal(prediction, cov)
            
        else: 
            sim_data = np.random.normal(prediction, data_unc)
            
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
    Class for fluospec model.
    Instantiate with parameters:
        line_prior_params: List of tuples of dicts of kwargs for gamma
                           priors for each spectral line, in the following order
                           A, w0, gamma, intensity_ratio
        {m,b}_prior_params: Dict of kwargs for m,b normal priors
        likelihood_type: str
            'normal': default, good if your noise is well-characterized
            'cauchy': helpful on data that has fine structure
            'mvnormal': for modeling data with covariances
        scale_prior_params: Dict of kwargs for scale trunc normal prior
                            Optional, only applies to 'cauchy' likelihood
        lkj_prior_params: Dict of kwargs for Cholesky LKJ prior
                          Optional, only applies to 'mvnormal'
                            
    
    Run `FluoSpecModel?` to see order of FluoSpecModel parameters. 
    
    '''
    line_prior_params: List[Tuple[Dict]]
    m_prior_params: Dict
    b_prior_params: Dict
    likelihood_type: str = 'cauchy'
    scale_prior_params: Dict = field(default_factory=dict)
    lkj_prior_params: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        valid_likelihoods = ['normal', 'cauchy', 'mvnormal']
        if self.likelihood_type not in valid_likelihoods:
            raise ValueError(f"{self.likelihood_type} is not a supported likelihood.")
    
    
    
    def model(self,
              spec_data_df: pd.DataFrame,
              cov: np.ndarray = None,
        ) -> pm.Model:
        """
        Builds generative model for fluorescence spectroscopy
        
        Parameters:
        -----------
        spec_data_df: pd.DataFrame
            dataframe of data to build model
        
        cov: np.ndarray
            Data covariance matrix. Only supported with mvnormal likelihood 
 
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
                                                       **A_params
                                                       ),
                                              
                                               pm.Gamma(f'w0_{i}',
                                                        **w0_params
                                                        ),
                                               pm.Gamma(f'gamma_{i}',
                                                        **gamma_params
                                                        ),
                                                pm.Gamma(f'intensity_ratio_{i}',
                                                         **intensity_params
                                                        ),
                                               )
                line_priors.append(specline)
                

            
            m = pm.Normal('m', **self.m_prior_params)
            b = pm.Normal('b', **self.b_prior_params)
            
            # convenient prior vector
            theta = (line_priors, m, b)
                
            # predicted intensity
            I_pred = pm.Deterministic('prediction',
                                      Prediction(*theta).prediction(w_data))
            

                
            if self.likelihood_type == 'normal':
                measurements = pm.Normal('I_model',
                                          mu=I_pred,
                                          sigma=sigma_I_data,
                                          observed=I_data
                                          )              

            elif self.likelihood_type == 'cauchy':
                # measurements modeled as Cauchian noise
                # accounts for broadening of Gaussian instrument noise
                # with undesired background fine structure
                
                scale = pm.TruncatedNormal('scale',
                                           **self.scale_prior_params
                                           )
                
                measurements = pm.Cauchy('I_model',
                                         alpha=I_pred,
                                         beta=scale,
                                         observed=I_data
                                         )
                
            # using mvnormal likelihood
            #
            # Note thtat this is largely experimental,
            # The api for this likelihood is unstable
            elif self.likelihood_type == 'mvnormal':
                dim = len(I_data)
                
                if self.covariance is not None:
                    measurements = pm.MvNormal('I_model',
                                               mu=I_pred,
                                               cov=self.covariance,
                                               observed=I_data
                                               )
                    
                else:
                    sd_dist = pm.Exponential.dist(sigma_I_data, shape=dim)
                    chol, _, _ = pm.LKJCholeskyCov('chol_cov',
                                                   n=dim,
                                                   sd_dist=sd_dist,
                                                   compute_corr=True,
                                                   **self.lkj_prior_params,
                                                   )
                    
                    measurements = pm.MvNormal('I_model',
                                               mu=I_pred,
                                               chol=chol,
                                               observed=I_data
                                               )
    
        return spectroscopy_model

    # TODO: think about extending above to allow multiple models,
    #       then create helper for model_comparison
    def model_comparison():
        raise NotImplementedError
        
def model_comparison(model1: FluoSpecModel,
                     model2: FluoSpecModel,
                     data: pd.DataFrame
    ) -> Tuple:
    '''
    

    Parameters
    ----------
    model1 : FluoSpecModel
        First model to compare.
    model2 : FluoSpecModel
        Second model to compare.
    data : pd.DataFrame
        Data to do inference.

    Returns
    -------
    Tuple
        Tuple containing odds ratio and model traces from smc sampler.

    '''
    with model1.model(data):
        trace_model1 = pm.sample_smc(2000)
        
    with model1.model(data):
        trace_model2 = pm.sample_smc(2000)
        
    odds_ratio = np.exp(trace_model1.report.log_marginal_likelihood - \
                        trace_model2.report.log_marginal_likelihood)
        
    return odds_ratio, trace_model1, trace_model2
        
    
        
    
        
        
    
    