import json
import numpy as np
import scipy.stats as ss
import CMR_IA as cmr
import time
import pandas as pd
import math
import pickle
import scipy.optimize as opt
import scipy.stats as st

def param_vec_to_dict(param_vec, sim_name):
    """
    Convert parameter vector to dictionary format expected by CMR2.
    """

    # Generate a base paramater dictionary
    param_dict = cmr.make_default_params()
    # param_dict.update(c_thresh = 0.4)

    # Put vector values into dict
    _, _, what_to_fit = make_boundary(sim_name)
    for name, value in zip(what_to_fit, param_vec):
        param_dict[name] = value

    return param_dict

def make_boundary(sim_name):
    """
    Make two vectors of boundary for parameters you want to fit.
    """

    # Generate a base paramater dictionary
    lb_dict = cmr.make_params()
    lb_dict.update(beta_enc = 0, 
                   beta_rec = 0, 
                   beta_cue = 0,
                   beta_distract = 0,
                   beta_rec_post = 0, 
                   phi_s = 0, 
                   phi_d = 0, 
                   s_cf = 0,
                   s_fc = 0,
                   kappa = 0, 
                   eta = 0, 
                   omega = 1, 
                   alpha = 0.5, 
                   c_thresh = 0,
                   c_thresh_itm = 0, 
                   c_thresh_ass = 0,
                   lamb = 0,
                   gamma_fc = 0, 
                   gamma_cf = 0,
                   d_ass = 0
                   )
    
    ub_dict = cmr.make_params()
    ub_dict.update(beta_enc = 1, 
                   beta_rec = 1,
                   beta_cue = 1,
                   beta_distract = 1, 
                   beta_rec_post = 1, 
                   phi_s = 8, 
                   phi_d = 5, 
                   s_cf = 1,
                   s_fc = 1,
                   kappa = 0.5, 
                   eta = 0.25, 
                   omega = 10, 
                   alpha = 1, 
                   c_thresh = 1,
                   c_thresh_itm = 1,
                   c_thresh_ass = 1, 
                   lamb = 0.25,
                   gamma_fc = 1, 
                   gamma_cf = 1,
                   d_ass = 1
                   )

   # Which Parameters to fit
    if sim_name == 'David':
        what_to_fit = ['beta_enc','beta_rec','beta_rec_post','s_fc','gamma_fc']
    elif sim_name == 'S1':
        what_to_fit = ['beta_enc', 'beta_rec', 'beta_cue', 'beta_rec_post', 'beta_distract', 'gamma_fc', 'gamma_cf', 's_fc', 's_cf', 'phi_s', 'phi_d', 'kappa', 'lamb', 'eta', 'omega', 'alpha', 'c_thresh', 'c_thresh_itm', 'c_thresh_ass', 'd_ass']
    elif sim_name == 'S2':
        what_to_fit = ['beta_enc', 'beta_rec', 'beta_cue', 'beta_rec_post', 'beta_distract', 'gamma_fc', 's_fc', 'c_thresh_itm', 'c_thresh_ass', 'd_ass']

    lb = [lb_dict[key] for key in what_to_fit]
    ub = [ub_dict[key] for key in what_to_fit]

    return lb, ub, what_to_fit