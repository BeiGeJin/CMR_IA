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
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.formula.api import logit
from pybeh.spc import spc
from pybeh.pfr import pfr
from pybeh.pli import pli
from pybeh.temp_fact import temp_fact
from pybeh.dist_fact import dist_fact
from pybeh.make_recalls_matrix import make_recalls_matrix
from pybeh.create_intrusions import intrusions
import statsmodels.formula.api as smf


def filter_by_condi(a, mods, prs, lls, dds, mod=None, pr=None, ll=None, dd=None):

    if pr == 's':
        pr = 1600
    elif pr == 'f':
        pr = 800

    ll = int(ll) if ll is not None else None
    dd = int(dd) if dd is not None else None

    ind = [i for i in range(len(a)) if ((ll is None or lls[i] == ll) and (pr is None or prs[i] == pr) and (mod is None or mods[i] == mod) and (dd is None or dds[i] == dd))]
    if len(ind) == 0:
        return np.array([])
    return np.array(a)[ind]


def pad_into_array(l, min_length=0):
    """
    Turn an array of uneven lists into a numpy matrix by padding shorter lists with zeros. Modified version of a
    function by user Divakar on Stack Overflow, here:
    http://stackoverflow.com/questions/32037893/numpy-fix-array-with-rows-of-different-lengths-by-filling-the-empty-elements-wi

    :param l: A list of lists
    :return: A numpy array made from l, where all rows have been made the same length via padding
    """
    l = np.array(l)
    # Get lengths of each row of data
    lens = np.array([len(i) for i in l])

    # If l was empty, we can simply return the empty numpy array we just created
    if len(lens) == 0:
        return lens

    # If all rows are the same length, return the original input as an array
    if lens.max() == lens.min() and lens.max() >= min_length:
        return l

    # Mask of valid places in each row
    mask = np.arange(max(lens.max(), min_length)) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=l.dtype)
    out[mask] = np.concatenate(l)

    return out


def get_data(data_files, wordpool_file, fixed_length=False):

    data_pres = np.empty((len(data_files), 16, 24), dtype='U32')  # Session x trial x serial position
    sources = np.zeros((len(data_files), 16, 24, 2))  # Session x trial x serial position x modality
    for i, data_file in enumerate(data_files):
        with open(data_file, 'r') as f:
            x = json.load(f)

        # Get words presented in this session (drop practice lists)
        data_pres[i, :, :] = np.array(x['pres_words'])[2:, :]
        # Get modality info
        for j, mod in enumerate(x['pres_mod'][2:]):
            sources[i, j, :, int(mod == 'v')] = 1

    # If fixed list length is activated, truncate lists to length 12 (or a given integer)
    if isinstance(fixed_length, int):
        if 0 < fixed_length <= 12:
            data_pres = data_pres[:, :, :fixed_length]
            sources = sources[:, :, :fixed_length, :]
        else:
            return ValueError('Invalid fixed list length. Must be between 1 and 12.')
    elif fixed_length == True:
        data_pres = data_pres[:, :, :12]
        sources = sources[:, :, :12, :]
    
    # Replace zeros with empty strings
    data_pres[data_pres == '0'] = ''

    # Get PEERS word pool
    wp = [s.lower() for s in np.loadtxt(wordpool_file, dtype='U32')]

    # Convert presented words to word ID numbers
    data_pres = np.searchsorted(wp, data_pres, side='right')
    
    # Create session indices
    sessions = []
    for n, sess_pres in enumerate(data_pres):
        sessions.append([n for _ in sess_pres])
    sessions = np.array(sessions)
    sessions = sessions.flatten()
    
    # Collapse sessions and trials of presented items into one dimension
    data_pres = data_pres.reshape((data_pres.shape[0] * data_pres.shape[1], data_pres.shape[2]))
    sources = sources.reshape((sources.shape[0] * sources.shape[1], sources.shape[2], sources.shape[3]))
        
    return data_pres, sessions, sources


def calc_spc(recalls, sessions, return_sem=False, listLength=12):

    s = spc(recalls, subjects=sessions, listLength=listLength)
    s_start = spc(recalls, subjects=sessions, listLength=listLength, start_position=[1])
    s_l4 = spc(recalls, subjects=sessions, listLength=listLength, start_position=listLength-np.arange(0, 4))

    if return_sem:
        return np.nanmean(s, axis=0), ss.sem(s, axis=0, nan_policy='omit'), \
               np.nanmean(s_start, axis=0), ss.sem(s_start, axis=0, nan_policy='omit'), \
               np.nanmean(s_l4, axis=0), ss.sem(s_l4, axis=0, nan_policy='omit')
    else:
        return s.mean(axis=0), np.nanmean(s_start, axis=0), np.nanmean(s_l4, axis=0)


def calc_pfr(recalls, sessions, return_sem=False, listLength=12):

    s = np.array(pfr(recalls, subjects=sessions, listLength=listLength))

    if return_sem:
        return s.mean(axis=0), ss.sem(s, axis=0)
    else:
        return s.mean(axis=0)


def calc_pli(intrusions, sessions, return_sem=False):

    s = np.array(pli(intrusions, subjects=sessions, per_list=True))

    if return_sem:
        return np.mean(s), ss.sem(s)
    else:
        return np.mean(s)


def pli_recency(intrusions, sessions, nmax=5, nskip=2, return_sem=False):
    
    u_sess = np.unique(sessions)
    n_sess = len(u_sess)

    result = np.zeros((n_sess, nmax))

    for i, sess in enumerate(u_sess):
        sess_intru = intrusions[sessions == sess]
        n_trials = len(sess_intru)
        pli_counts = np.zeros(n_trials-1)
        possible_counts = np.arange(n_trials-1, 0, -1)
        for trial, trial_data in enumerate(sess_intru):
            if trial < nskip:
                continue
            for item in trial_data:
                if item > 0:
                    pli_counts[item-1] += 1
        normed_counts = pli_counts / possible_counts
        result[i, :] = normed_counts[:nmax] / np.nansum(normed_counts)

    if return_sem:
        return np.nanmean(result, axis=0), ss.sem(result, axis=0, nan_policy='omit')
    else:
        return np.nanmean(result, axis=0)

    
def calc_temp_fact(recalls, sessions, listLength=12, skip_first_n=0, return_sem=False):
    
    s = temp_fact(recalls, sessions, listLength=listLength, skip_first_n=skip_first_n)
    
    if return_sem:
        return np.nanmean(s, axis=0), ss.sem(s, axis=0, nan_policy='omit')
    else:
        return np.nanmean(s, axis=0)


def calc_sem_fact(rec_nos, pres_nos, sessions, dist_mat, skip_first_n=0, return_sem=False):
    
    s = dist_fact(rec_nos, pres_nos, sessions, dist_mat, is_similarity=True, skip_first_n=skip_first_n)
    
    if return_sem:
        return np.nanmean(s, axis=0), ss.sem(s, axis=0, nan_policy='omit')
    else:
        return np.nanmean(s, axis=0)
    

def param_vec_to_dict(param_vec):
    """
    Convert parameter vector to dictionary format expected by CMR2.
    """
    # Generate a base paramater dictionary
    param_dict = cmr.make_default_params()
    param_dict.update(c_thresh = 0.4) ###!!!
#     param_dict.update(beta_enc = 0.5, 
#                beta_rec = 0.5, 
#                beta_rec_post = 0.5, 
#                phi_s = 2, 
#                phi_d = 0.5, 
#                s_cf = 0,
#                s_fc = 0,
#                kappa = 0.5, 
#                eta = 0.5, 
#                omega = 8, 
#                alpha = 4, 
#                c_thresh = 0.5, 
#                lamb = 0.5,
#                gamma_fc = 0.5, 
#                gamma_cf = 0.5)
    
    _, _, what_to_fit = make_boundary()
    
    for i in range(len(param_vec)):
        param_dict[what_to_fit[i]] = param_vec[i]

    return param_dict

def make_boundary():
    """
    Make two vectors of boundary for parameters you want to fit.
    """
    # Generate a base paramater dictionary
    lb_dict = cmr.make_params()
    lb_dict.update(beta_enc = 0.1, 
                   beta_rec = 0.1, 
                   beta_rec_post = 0.01, 
                   phi_s = None, 
                   phi_d = None, 
                   s_cf = None,
                   s_fc = 0.01,
                   kappa = None, 
                   eta = None, 
                   omega = None, 
                   alpha = None, 
                   c_thresh = None, 
                   lamb = None,
                   gamma_fc = 0.01, 
                   gamma_cf = None,
                   a = None,
                   b = None)
    
    ub_dict = cmr.make_params()
    ub_dict.update(beta_enc = 0.9, 
                   beta_rec = 0.9, 
                   beta_rec_post = 0.99, 
                   phi_s = None, 
                   phi_d = None, 
                   s_cf = None,
                   s_fc = 1,
                   kappa = None, 
                   eta = None, 
                   omega = None, 
                   alpha = None, 
                   c_thresh = None, 
                   lamb = None,
                   gamma_fc = 0.99, 
                   gamma_cf = None,
                   a = None,
                   b = None)
    
    what_to_fit = ['beta_enc','beta_rec','beta_rec_post','s_fc','gamma_fc']
    lb = [lb_dict[key] for key in what_to_fit]
    ub = [ub_dict[key] for key in what_to_fit]

    return lb, ub, what_to_fit

def obj_func(param_vec, df, w2v, sources, return_recalls=False, mode='RT_score'):
    
    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec)
    
    
    # Run model with the parameters given in param_vec
    df_simu = cmr.run_continuous_recog_multi_sess(param_dict, df, w2v)
    df_simu = df_simu.merge(df, on=['session','position','itemno'])
    
#     subjectlist = np.unique(df.subject_ID)
#     df['s_resp'] = np.repeat(np.nan, len(df))
#     df['s_rt'] = np.repeat(np.nan, len(df))
    
#     for subj in subjectlist:
        
#         start_time = time.time()
        
#         pres_nos = df.loc[df.subject_ID == subj, 'itemno'].to_list()
#         cue_mat = pres_nos
#         p_mat = np.reshape(pres_nos,(len(pres_nos),1))
#         model = cmr.CMR2(params = param_dict, pres_mat = p_mat, sem_mat = w2v, 
#                          cue_mat = cue_mat, task = 'Recog', mode = 'Continuous')
#         model.run_continuous_recog_single_sess()
        
#         recs = model.rec_items
#         rts = model.rec_times
#         csim = model.recog_similarity
        
#         df.loc[df.subject_ID == subj, 's_resp'] = recs
#         df.loc[df.subject_ID == subj, 's_rt'] = rts
        
#         print("--- %s seconds ---" % (time.time() - start_time))
    # print(df)
    
    # Score the model's behavioral stats as compared with the true data    
    if mode == 'McNemar':
        err = McNemar_chi_square(df_simu, 'yes', 's_resp')
    elif mode == 'Deviance':
        err = mean_deviance(df_simu, 'yes', 's_resp')
    elif mode == 'Logit':
        err = logit_negloglikelihood(df_simu, 'yes', 'csim')
    elif mode == 'HitFa':
        err = hit_fa(df_simu, 'yes', 's_resp')
    elif mode == 'chi_squared':
        err = chi_squared(df_simu, 'yes', 's_resp')
    elif mode == 'ML':
        mu, sig, err = twopeak_mle(df_simu)
    elif mode == 'RT':
        err = rt_rmse(df_simu)
    elif mode == 'RT_score':
        err = rt_score(df_simu)
    
    cmr_stats = {}
    cmr_stats['err'] = err
    cmr_stats['params'] = param_vec
    
    if mode == 'ML':
        cmr_stats['curve'] = (mu, sig)
    
    if return_recalls:
        return err, cmr_stats, df_simu
    else:
        return err, cmr_stats

def McNemar_chi_square(df, target_col, cmr_col):
    
    cont_table = pd.crosstab(df[target_col], df[cmr_col])
    
    if cont_table.shape != (2,2):
        return np.nan_to_num(np.inf)
    
    else: 
        result = mcnemar(cont_table, exact=False)
        chi2_err = result.statistic

        if math.isinf(chi2_err):
            chi2_err = 0

        return chi2_err

def mean_deviance(df, target_col, cmr_col):
    
    md = (df[target_col] - df[cmr_col]).abs().mean(axis=0,skipna=True)
    
    return md

def logit_negloglikelihood(df, target_col, cmr_col):
    
    try:
        formula = target_col + " ~ " + cmr_col
        log_reg = logit(formula, data=df).fit()
        neg_mll = -log_reg.llf/log_reg.nobs

        return neg_mll
    
    except:
        
        print("singular?")
        with open('df_singular.pkl', 'wb') as outp:
            pickle.dump(df, outp, pickle.HIGHEST_PROTOCOL)
            
        return 999999
    
def hit_fa(df, target_col, cmr_col):
    
    df = df.loc[pd.notna(df.yes)].reset_index()
    df = df.astype({'yes': 'int32'})
    
    hit_r = df.loc[df.old == True].groupby(['subject_ID'])[target_col].mean().mean()
    fa_r = df.loc[df.old == False].groupby(['subject_ID'])[target_col].mean().mean()
    hit_s = df.loc[df.old == True][cmr_col].mean()
    fa_s = df.loc[df.old == False][cmr_col].mean()
    
    rmse = math.sqrt(((hit_r - hit_s)**2 + (fa_r - fa_s)**2)/2)

    return rmse

def chi_squared(df, target_col, cmr_col):
    
    df = df.loc[pd.notna(df.yes)].reset_index()
    df = df.astype({'yes': 'int32'})
    
    df = df.loc[df.lag > 0]
    df = df.assign(lag_bin = df['lag'] // 10 * 10)
    df['log_lag'] = np.log(df['lag'])
    df['log_lag_bin'] = pd.cut(df['log_lag'], np.arange(df['log_lag'].max()+1), labels=False, right=False)
    df = df.loc[df.log_lag_bin<5]
    # df = df.loc[df.lag < 110]
    
    df_laggp = df.groupby(['subject_ID','log_lag_bin'])[target_col].mean()
    
    df_laggp = df_laggp.to_frame(name='hr').reset_index()
    df_laggp = df_laggp.groupby(['log_lag_bin']).agg({'hr':['mean','sem']})
    df_laggp.columns = df_laggp.columns.to_flat_index().map(lambda x: '_'.join(x))
    df_laggp = df_laggp.reset_index()
    
    y = df_laggp['hr_mean'].to_numpy()
    y_sem = df_laggp['hr_sem'].to_numpy()
    y_hat = df.groupby(['log_lag_bin'])[cmr_col].mean().to_numpy()
    
    chi2_err = np.mean(((y - y_hat) / y_sem) ** 2)

    return chi2_err

def gauss_logL(mu, sig, arr):
    
    N = len(arr)
    ll = - 0.5 * N * math.log(2*math.pi) - N * math.log(sig) - 0.5 * np.sum((((arr-mu)/sig)**2))
    
    return ll

def twopeak_logL(params, df):
    
    k = 1
    d = k*1.5
    
    mu, sig = params
    
    df_new = df.loc[df.old == False]
    csim_new = df_new.csim.to_numpy()
    logL = gauss_logL(mu, sig, csim_new)
    
    df_old = df.loc[df.old == True]
    csim_old = df_old.csim.to_numpy()
    logL += gauss_logL(mu + d*sig, k*sig, csim_old)
    
    return logL

def twopeak_mle(df):
    
    df = df.loc[pd.notna(df.yes)].reset_index()
    df = df.astype({'yes': 'int32'})
    
    res = opt.minimize(
        fun=lambda params, df: -twopeak_logL(params, df),
        x0=np.array([0.8, 0.8]), args=(df,),bounds = ((0.001,1),(0.001,1)), method='Nelder-Mead')
    
    mu, sig = res.x
    
    return mu, sig, res.fun

def rt_rmse(df):
    
    df = df.loc[(df.rt < 3000) & (df.rt > 400)]
    rmse = np.sqrt(np.mean((df.s_rt - df.rt)**2))
    
    return rmse

def loftus_masson(df, sub_cols, cond_col, value_col, within_cols=[]):
    
    if not isinstance(sub_cols, list):
        sub_cols = [sub_cols]
    if not isinstance(within_cols, list):
        within_cols = [within_cols]
    df = df.copy()
    if len(within_cols) > 0:
        df['M'] = df.groupby(within_cols)[value_col].transform('mean')
    else:
        df['M'] = df[value_col].mean()
    df['M_S'] = df.groupby(sub_cols + within_cols)[value_col].transform('mean')
    df['adj_' + value_col] = (df[value_col] + df['M'] - df['M_S'])
    
    return df

def rt_score(df):
    
    df = df.loc[(df.rt < 3000) & (df.rt > 400) & (df.position > 10) & (df.old == df.yes)]
    df = loftus_masson(df, 'subject_ID', [], 'rt')
    
    a = 2800
    c_thresh = 0.4
    df['csim_diff'] = df['csim'] - c_thresh
    df['csim_score'] = np.power(-1, df['yes']) * (np.log(df.adj_rt) - np.log(a)) 
    df.csim_score = df.csim_score.astype("float")
    mod = smf.ols(formula='csim_score ~ -1 + csim_diff', data=df)
    res = mod.fit()
    mse = res.mse_resid
    
    return mse
    

# def chi_squared_error(target_stats, cmr_stats):
    
#     y = []
#     y_sem = []
#     y_hat = []
    
#     # Fit SPC and PFR
#     for stat in ('spc_fr1', 'spc_frl4', 'pfr'):
#         for ll in cmr_stats[stat]:
#             # Skip serial position 1 for SPC when initiating recall from position 1 
#             # (to avoid dividing by 0 standard error, since prec is always 1 by definition)
#             if stat == 'spc_fr1':
#                 y.append(np.atleast_1d(target_stats[stat][ll][1:]))
#                 y_sem.append(np.atleast_1d(target_stats[stat + '_sem'][ll][1:]))
#                 y_hat.append(np.atleast_1d(cmr_stats[stat][ll][1:]))
#             else:
#                 y.append(np.atleast_1d(target_stats[stat][ll]))
#                 y_sem.append(np.atleast_1d(target_stats[stat + '_sem'][ll]))
#                 y_hat.append(np.atleast_1d(cmr_stats[stat][ll]))
    
#     # Fit PLIs and PLI recency (not separated by list length)
#     for stat in ('plis', 'pli_recency'):
#         y.append(np.atleast_1d(target_stats[stat]))
#         y_sem.append(np.atleast_1d(target_stats[stat + '_sem']))
#         y_hat.append(np.atleast_1d(cmr_stats[stat]))
        
#     y = np.concatenate(y)
#     y_sem = np.concatenate(y_sem)
#     y_hat = np.concatenate(y_hat)
    
#     chi2_err = np.mean(((y - y_hat) / y_sem) ** 2)
    
#     return chi2_err


# def sim1c_error(target_stats, cmr_stats):
#     """
#     Sim 1c fits only the conditional SPCs and PFR, and uses mean squared error 
#     instead of chi-squared error because standard errors are not available.
#     """
#     y = []
#     y_hat = []
    
#     # Fit SPC and PFR
#     for stat in ('spc_fr1', 'spc_frl4', 'pfr'):
#         for ll in cmr_stats[stat]:
#             # Skip serial position 1 for SPC when initiating recall from position 1 
#             if stat == 'spc_fr1':
#                 y.append(np.atleast_1d(target_stats[stat][ll][1:]))
#                 y_hat.append(np.atleast_1d(cmr_stats[stat][ll][1:]))
#             else:
#                 y.append(np.atleast_1d(target_stats[stat][ll]))
#                 y_hat.append(np.atleast_1d(cmr_stats[stat][ll]))
        
#     y = np.concatenate(y)
#     y_hat = np.concatenate(y_hat)
    
#     mse = np.mean((y - y_hat) ** 2)

#     return mse