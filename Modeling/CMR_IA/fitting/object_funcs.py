import numpy as np
import scipy.stats as ss
import CMR_IA as cmr
import time
import pandas as pd
import math
import pickle
import scipy as sp
from optimization_utils import param_vec_to_dict

def obj_func_S1(param_vec, df_study, df_test, sem_mat, sources, return_df=False):
    
    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name = 'S1')
    stats = []

    for i in [1,2,3]:
        # Separate 3 groups of simulation
        df_study_gp = df_study.query(f"group == {i}").copy()
        df_test_gp = df_test.query(f"group == {i}").copy()

        if i == 1:  # asso-CR
            nitems = 4 * 48 # 96
            test1_num = 40
        elif i == 2:  # pair-CR
            nitems = 4 * 48 # 176
            test1_num = 80
        elif i == 3:  # item-CR
            nitems = 4 * 48 # 136
            test1_num = 80

        # Run model with the parameters given in param_vec
        param_dict.update(nitems_in_accumulator = nitems, learn_while_retrieving = True, rec_time_limit=10000)
        df_simu, _, _ = cmr.run_success_multi_sess(param_dict, df_study_gp, df_test_gp, sem_mat)
        # print(df_simu)
        # print(df_test_gp)
        df_simu['test'] = df_test_gp['test']
        df_simu = df_simu.merge(df_test_gp,on=['session','test','test_itemno1','test_itemno2'])

        # Get behavioral stats
        subjects = np.unique(df_simu.subject)
        stats_gp = []
        for subj in subjects:
            df_subj = df_simu.query(f"subject == {subj}").copy()
            stats_gp.append(list(anal_perform_S1(df_subj)))
        stats_mean = np.mean(stats_gp, axis=0)
        stats.append(list(stats_mean))
    
    # Score the model's behavioral stats as compared with the true data    
    stats = np.array(stats)
    ground_truth = np.array([[0.42, 0.72, 0.22, 0.81], [0.30, 0.80, 0.12, 0.71], [0.19, 0.67, 0.15, 0.57]])  # p_rc, hr, far, q
    err = np.mean(np.power(stats - ground_truth,2))
    
    cmr_stats = {}
    cmr_stats['err'] = err
    cmr_stats['params'] = param_vec
    cmr_stats['stats'] = stats
    
    if return_df:
        return err, cmr_stats, df_simu
    else:
        return err, cmr_stats


def anal_perform_S1(df_simu):

    # Get correctness
    df_simu['correct'] = df_simu.s_resp == df_simu.correct_ans

    # Recognition performance
    df_recog = df_simu.query("test==1")
    hr_far = df_recog.groupby("correct_ans")["s_resp"].mean().to_frame(name="Yes rate")
    hr = hr_far['Yes rate'][1]
    far = hr_far['Yes rate'][0]
    # print("recognition: \n", hr_far)

    # Cued recall performance
    df_cr = df_simu.query("test==2")
    p_rc = df_cr.correct.mean()
    # print("cued recall: \n", p_rc)

    # Analyze pair
    def get_pair(df_tmp):
        df_tmp_pair = pd.pivot_table(df_tmp,index="pair_idx",columns="test",values="correct")
        df_tmp_pair.columns = ["test1","test2"]
        df_tmp_pair.reset_index(inplace=True)
        return df_tmp_pair
    df_simu_p = df_simu.query("pair_idx >= 0")
    df_pair = df_simu_p.groupby("session").apply(get_pair).reset_index()
    test2_rsp = pd.Categorical(df_pair.test2, categories=[0,1])
    test1_rsp = pd.Categorical(df_pair.test1, categories=[0,1])
    df_tab = pd.crosstab(index=test2_rsp,columns=test1_rsp, rownames=['test2'], colnames=['test1'], normalize=False, dropna=False)
    # print("contingency table: \n", df_tab)

    # Compute Q values
    def Yule_Q(A, B, C, D):
        return (A * D - B * C) / (A * D + B * C)
    q = Yule_Q(df_tab[1][1]+0.5,df_tab[0][1]+0.5,df_tab[1][0]+0.5,df_tab[0][0]+0.5)  # add 0.5
    # print("Q: ", q)

    return p_rc, hr, far, q

def obj_func_S2(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name = 'S2')
    
    # Run model with the parameters given in param_vec
    param_dict.update(learn_while_retrieving=True)
    df_simu, _, _ = cmr.run_success_multi_sess(param_dict, df_study, df_test, sem_mat, mode="Recog-Recog")
    df_simu['test'] = df_test['test']
    df_simu = df_simu.merge(df_test,on=['session','list','test','test_itemno1','test_itemno2'])

    # Get correctness
    df_simu['correct'] = df_simu.s_resp == df_simu.correct_ans

    # Get conditions
    def get_cond(x):
        this_type = x['type']
        target = x['correct_ans']
        if target == 1:
            if this_type == "Different_Item":
                return "Different_Item"
            elif this_type == "Item_Pair":
                return "Item_Pair"
            elif this_type == "Pair_Item":
                return "Pair_Item"
            elif this_type == "Same_Item":
                return "Same_Item"
            elif this_type == "Intact_Pair":
                return "Intact_Pair"
        elif target == 0:
            if this_type == "extra":
                return "NR_Lure"
            elif this_type == "Same_Item" or this_type == "Intact_Pair":
                return "Repeated_Lure"
            else:
                return "Discard"
    df_simu['condition'] = df_simu.apply(get_cond,axis=1)

    # Get behavioral stats
    subjects = np.unique(df_simu.subject)
    stats = []
    for subj in subjects:
        df_subj = df_simu.query(f"subject=={subj} and list % 3 != 0") # discard first list
        # df_subj = df_simu.query(f"subject=={subj}")
        stats_subj = anal_perform_S2(df_subj)
        stats.append(stats_subj)

    # Score the model's behavioral stats as compared with the true data  
    stats_mean = np.nanmean(stats, axis=0)
    ground_truth = np.array([[0.82, 0.68, 0.26], 
                         [0.82, 0.85, 0.64],
                         [0.91, 0.85, 0.59],
                         [0.81, 0.82, 0.86],
                         [0.90, 0.92, 0.94],
                         [0.07, 0.15, 0.54],
                         [0.07, 0.06, 0]])
    err = np.mean(np.power(stats_mean-ground_truth,2))

    cmr_stats = {}
    cmr_stats['err'] = err
    cmr_stats['params'] = param_vec
    cmr_stats['stats'] = stats_mean

    return err, cmr_stats


def anal_perform_S2(df_simu):
    
    # Get target items
    df_target = df_simu.query("condition != 'Discard'")

    # Get pairs data
    def get_pair(df_tmp):
        df_tmp_pair = pd.pivot_table(df_tmp,index=["pair_idx","condition"],columns="test",values="correct")
        df_tmp_pair.columns = ["test1","test2"]
        df_tmp_pair.reset_index(inplace=True)
        return df_tmp_pair

    df_p = df_target.query("condition != 'NR_Lure'")
    df_pair = get_pair(df_p).reset_index()

    # Get Q values
    def Yule_Q(A, B, C, D):
        return (A * D - B * C) / (A * D + B * C)
    
    qs = []
    conditions = ['Different_Item', 'Item_Pair', 'Pair_Item', 'Same_Item', 'Intact_Pair', 'Repeated_Lure', 'NR_Lure']
    for cond in conditions:
        df_tmp = df_pair.query(f"condition == '{cond}'")
        test2_rsp = pd.Categorical(df_tmp.test2, categories=[0,1])
        test1_rsp = pd.Categorical(df_tmp.test1, categories=[0,1])
        df_tab = pd.crosstab(index=test2_rsp,columns=test1_rsp, rownames=['test2'], colnames=['test1'], normalize=False, dropna=False)

        try:
            q = Yule_Q(df_tab[1][1]+0.5,df_tab[0][1]+0.5,df_tab[1][0]+0.5,df_tab[0][0]+0.5)
        except:
            q = 0 if cond == 'NR_Lure' else np.nan

        qs.append(q)
    
    # Get hit rates and aggregate
    df_res = pd.DataFrame({'Condition':conditions, 'Q':qs})
    df_res.set_index('Condition', inplace=True)
    df_res['Test1_p'] = df_target.groupby(["test","condition"])["s_resp"].mean()[1]
    df_res['Test2_p'] = df_target.groupby(["test","condition"])["s_resp"].mean()[2]
    df_res = df_res[['Test1_p','Test2_p','Q']]
    stats = df_res.values.tolist()

    return stats


def obj_func_6b(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name = '6b')
    
    # Run model with the parameters given in param_vec
    param_dict.update(learn_while_retrieving=True, nitems_in_accumulator = 96, use_new_context = True)
    df_simu, _, _ = cmr.run_success_multi_sess(param_dict, df_study, df_test, sem_mat, mode="CR-CR")
    df_simu['test_pos'] = np.tile(np.arange(1,25),600)  # 100 * 6
    df_simu = df_simu.merge(df_test,on=['session','list', 'test_itemno1','test_itemno2', 'test_pos'])

    # Get correctness
    df_simu['correct'] = df_simu.s_resp == df_simu.correct_ans

    # Get conditions
    df_cond = df_simu.groupby(["pair_idx","test"])['order'].mean().to_frame(name='corr_rate').reset_index()
    df_cond = df_cond.pivot_table(index='pair_idx',columns='test',values='corr_rate').reset_index()
    df_cond.columns = ['pair_idx','test1','test2']

    def cond(x):
        test1 = x['test1']
        test2 = x['test2']
        if test1 == 1 and test2 == 1:
            return 'F-F'
        elif test1 == 1 and test2 == 2:
            return 'F-B'
        elif test1 == 2 and test2 == 1:
            return 'B-F'
        elif test1 == 2 and test2 == 2:
            return 'B-B'

    df_cond['cond'] = df_cond.apply(lambda x:cond(x),axis=1)
    df_cond['cong'] = df_cond.apply(lambda x: 'Identical' if x['cond'] == 'F-F' or x['cond'] == 'B-B' else 'Reversed',axis=1)
    pairidx2cond = df_cond.loc[:,['pair_idx','cond']].set_index("pair_idx").to_dict()['cond']
    pairidx2cong = df_cond.loc[:,['pair_idx','cong']].set_index("pair_idx").to_dict()['cong']
    df_simu['cond'] = df_simu.apply(lambda x:pairidx2cond[x['pair_idx']],axis=1)
    df_simu['cong'] = df_simu.apply(lambda x:pairidx2cong[x['pair_idx']],axis=1)

    # Get behavioral stats
    subjects = np.unique(df_simu.session)
    inde_stats = []
    reve_stats = []
    for subj in subjects:
        df_subj_inde = df_simu.query(f"session == {subj} and cong == 'Identical'").copy()
        inde_stats.append(list(anal_perform_6b(df_subj_inde)))

        df_subj_reve = df_simu.query(f"session == {subj} and cong == 'Reversed'").copy()
        reve_stats.append(list(anal_perform_6b(df_subj_reve)))

    # Score the model's behavioral stats as compared with the true data 
    inde_stats_mean = np.mean(inde_stats,axis=0)
    reve_stats_mean = np.mean(reve_stats,axis=0)
    inde_ground_truth = np.array([0.319, 0.006, 0.012, 0.663, 0.94])
    reve_ground_truth = np.array([0.293, 0.049, 0.122, 0.537, 0.96])
    err = (np.mean(np.power(inde_stats_mean-inde_ground_truth,2)) + np.mean(np.power(reve_stats_mean-reve_ground_truth,2)))/2

    cmr_stats = {}
    cmr_stats['err'] = err
    cmr_stats['params'] = param_vec
    cmr_stats['stats'] = [inde_stats_mean, reve_stats_mean]

    return err, cmr_stats

def anal_perform_6b(df_simu):

    # get pair
    df_pair = pd.pivot_table(df_simu,index='pair_idx',columns='test', values= 'correct')
    df_pair.columns = ['test1','test2']
    test2_rsp = pd.Categorical(df_pair.test2, categories=[1,0])
    test1_rsp = pd.Categorical(df_pair.test1, categories=[1,0])
    df_tab = pd.crosstab(index=test2_rsp,columns=test1_rsp, rownames=['test2'], colnames=['test1'], normalize=False, dropna=False)
    df_tab_norm = pd.crosstab(index=test2_rsp,columns=test1_rsp, rownames=['test2'], colnames=['test1'], normalize='all', dropna=False)
    t1_t2 = df_tab_norm[1][1] # 1, 2
    t1_f2 = df_tab_norm[1][0]
    f1_t2 = df_tab_norm[0][1]
    f1_f2 = df_tab_norm[0][0]
    # print(df_tab)
    # print(df_tab_norm)
    # print(t1_t2, t1_f2, f1_t2, f1_f2)

    # compute" Q
    def Yule_Q(A, B, C, D):
        return (A * D - B * C) / (A * D + B * C)
    q = Yule_Q(df_tab[1][1]+0.5,df_tab[0][1]+0.5,df_tab[1][0]+0.5,df_tab[0][0]+0.5)  # add 0.5
    # print("Q: ", q)

    return t1_t2, t1_f2, f1_t2, f1_f2, q

def obj_func_3(param_vec, df_study, df_test, sem_mat, sources):

    assert df_study == None
    df = df_test

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name = '3')
    param_dict.update(use_new_context = True)

    # Run model with the parameters given in param_vec
    df_simu = cmr.run_conti_recog_multi_sess(param_dict, df, sem_mat, mode="Hockley")
    df_simu = df_simu.merge(df,on=['session','position','study_itemno1','study_itemno2','test_itemno1','test_itemno2'])

    # group by type and lag
    df_laggp = df_simu.groupby(['type','lag']).s_resp.mean().to_frame(name='yes_rate').reset_index()

    # get d prime
    # df_dprime = pd.DataFrame()
    # df_dprime['lag'] = [2,4,6,8,16]
    # df_dprime['I_z_hr'] = sp.stats.norm.ppf(df_laggp.loc[df_laggp.type == 1, 'yes_rate'].astype(float))
    # df_dprime['I_z_far'] = np.mean(sp.stats.norm.ppf(df_laggp.loc[df_laggp.type == 0, 'yes_rate'].astype(float)))
    # df_dprime['I_dprime'] = df_dprime['I_z_hr'] - df_dprime['I_z_far']
    # df_dprime['A_z_hr'] = sp.stats.norm.ppf(df_laggp.loc[df_laggp.type == 2, 'yes_rate'].astype(float))
    # df_dprime['A_z_far'] = sp.stats.norm.ppf(df_laggp.loc[df_laggp.type == 3, 'yes_rate'].astype(float))
    # df_dprime['A_dprime'] = df_dprime['A_z_hr'] - df_dprime['A_z_far']

    # get the vectors
    I_hr = df_laggp.loc[df_laggp.type == 'single_old', "yes_rate"].to_numpy()
    I_far = np.mean(df_laggp.loc[df_laggp.type == 'single_new', 'yes_rate'].astype(float))
    A_hr = df_laggp.loc[df_laggp.type == 'pair_old', "yes_rate"].to_numpy()
    A_far = df_laggp.loc[df_laggp.type == 'pair_new', "yes_rate"].to_numpy()

    # ground truth
    I_hr_gt = np.array([0.865, 0.811, 0.752, 0.746, 0.708])
    I_far_gt = 0.15 # 0.12
    A_hr_gt = np.array([0.843, 0.787, 0.720, 0.735, 0.646])
    A_far_gt = np.array([0.406, 0.371, 0.285, 0.259, 0.202])

    # calculate the error
    err = np.mean(np.power(I_hr - I_hr_gt, 2)) + np.mean(np.power(A_hr - A_hr_gt, 2)) \
        + np.power(I_far - I_far_gt, 2) * 5 + np.mean(np.power(A_far - A_far_gt, 2))
    
    # apply some constraints
    if not (I_hr[0] > I_hr[1] and I_hr[1] > I_hr[2] and I_hr[2] > I_hr[3] and I_hr[3] > I_hr[4]):
        err += 1
    if not (A_hr[0] > A_hr[1] and A_hr[1] > A_hr[2] and A_hr[2] > A_hr[3] and A_hr[3] > A_hr[4]):
        err += 1
    if not (I_hr > A_hr).all():
        err += 1
    if not (A_far[0] > A_far[1] and A_far[1] > A_far[2] and A_far[2] > A_far[3] and A_far[3] > A_far[4]):
        err += 1 

    cmr_stats = {}
    cmr_stats['err'] = err
    cmr_stats['params'] = param_vec
    cmr_stats['stats'] = [I_hr, I_far, A_hr, A_far]

    return err, cmr_stats


# def obj_func(param_vec, df, w2v, sources, return_recalls=False, mode='RT_score'):
    
#     # Reformat parameter vector to the dictionary format expected by CMR2
#     param_dict = param_vec_to_dict(param_vec)
    
#     # Run model with the parameters given in param_vec
#     df_simu = cmr.run_continuous_recog_multi_sess(param_dict, df, w2v)
#     df_simu = df_simu.merge(df, on=['session','position','itemno'])
    
#     # Score the model's behavioral stats as compared with the true data    
#     if mode == 'McNemar':
#         err = McNemar_chi_square(df_simu, 'yes', 's_resp')
#     elif mode == 'Deviance':
#         err = mean_deviance(df_simu, 'yes', 's_resp')
#     elif mode == 'Logit':
#         err = logit_negloglikelihood(df_simu, 'yes', 'csim')
#     elif mode == 'HitFa':
#         err = hit_fa(df_simu, 'yes', 's_resp')
#     elif mode == 'chi_squared':
#         err = chi_squared(df_simu, 'yes', 's_resp')
#     elif mode == 'ML':
#         mu, sig, err = twopeak_mle(df_simu)
#     elif mode == 'RT':
#         err = rt_rmse(df_simu)
#     elif mode == 'RT_score':
#         err = rt_score(df_simu)
    
#     cmr_stats = {}
#     cmr_stats['err'] = err
#     cmr_stats['params'] = param_vec
    
#     if mode == 'ML':
#         cmr_stats['curve'] = (mu, sig)
    
#     if return_recalls:
#         return err, cmr_stats, df_simu
#     else:
#         return err, cmr_stats

# def McNemar_chi_square(df, target_col, cmr_col):
    
#     cont_table = pd.crosstab(df[target_col], df[cmr_col])
    
#     if cont_table.shape != (2,2):
#         return np.nan_to_num(np.inf)
    
#     else: 
#         result = mcnemar(cont_table, exact=False)
#         chi2_err = result.statistic

#         if math.isinf(chi2_err):
#             chi2_err = 0

#         return chi2_err

# def mean_deviance(df, target_col, cmr_col):
    
#     md = (df[target_col] - df[cmr_col]).abs().mean(axis=0,skipna=True)
    
#     return md

# def logit_negloglikelihood(df, target_col, cmr_col):
    
#     try:
#         formula = target_col + " ~ " + cmr_col
#         log_reg = logit(formula, data=df).fit()
#         neg_mll = -log_reg.llf/log_reg.nobs

#         return neg_mll
    
#     except:
        
#         print("singular?")
#         with open('df_singular.pkl', 'wb') as outp:
#             pickle.dump(df, outp, pickle.HIGHEST_PROTOCOL)
            
#         return 999999
    
# def hit_fa(df, target_col, cmr_col):
    
#     df = df.loc[pd.notna(df.yes)].reset_index()
#     df = df.astype({'yes': 'int32'})
    
#     hit_r = df.loc[df.old == True].groupby(['subject_ID'])[target_col].mean().mean()
#     fa_r = df.loc[df.old == False].groupby(['subject_ID'])[target_col].mean().mean()
#     hit_s = df.loc[df.old == True][cmr_col].mean()
#     fa_s = df.loc[df.old == False][cmr_col].mean()
    
#     rmse = math.sqrt(((hit_r - hit_s)**2 + (fa_r - fa_s)**2)/2)

#     return rmse

# def chi_squared(df, target_col, cmr_col):
    
#     df = df.loc[pd.notna(df.yes)].reset_index()
#     df = df.astype({'yes': 'int32'})
    
#     df = df.loc[df.lag > 0]
#     df = df.assign(lag_bin = df['lag'] // 10 * 10)
#     df['log_lag'] = np.log(df['lag'])
#     df['log_lag_bin'] = pd.cut(df['log_lag'], np.arange(df['log_lag'].max()+1), labels=False, right=False)
#     df = df.loc[df.log_lag_bin<5]
#     # df = df.loc[df.lag < 110]
    
#     df_laggp = df.groupby(['subject_ID','log_lag_bin'])[target_col].mean()
    
#     df_laggp = df_laggp.to_frame(name='hr').reset_index()
#     df_laggp = df_laggp.groupby(['log_lag_bin']).agg({'hr':['mean','sem']})
#     df_laggp.columns = df_laggp.columns.to_flat_index().map(lambda x: '_'.join(x))
#     df_laggp = df_laggp.reset_index()
    
#     y = df_laggp['hr_mean'].to_numpy()
#     y_sem = df_laggp['hr_sem'].to_numpy()
#     y_hat = df.groupby(['log_lag_bin'])[cmr_col].mean().to_numpy()
    
#     chi2_err = np.mean(((y - y_hat) / y_sem) ** 2)

#     return chi2_err

# def gauss_logL(mu, sig, arr):
    
#     N = len(arr)
#     ll = - 0.5 * N * math.log(2*math.pi) - N * math.log(sig) - 0.5 * np.sum((((arr-mu)/sig)**2))
    
#     return ll

# def twopeak_logL(params, df):
    
#     k = 1
#     d = k*1.5
    
#     mu, sig = params
    
#     df_new = df.loc[df.old == False]
#     csim_new = df_new.csim.to_numpy()
#     logL = gauss_logL(mu, sig, csim_new)
    
#     df_old = df.loc[df.old == True]
#     csim_old = df_old.csim.to_numpy()
#     logL += gauss_logL(mu + d*sig, k*sig, csim_old)
    
#     return logL

# def twopeak_mle(df):
    
#     df = df.loc[pd.notna(df.yes)].reset_index()
#     df = df.astype({'yes': 'int32'})
    
#     res = opt.minimize(
#         fun=lambda params, df: -twopeak_logL(params, df),
#         x0=np.array([0.8, 0.8]), args=(df,),bounds = ((0.001,1),(0.001,1)), method='Nelder-Mead')
    
#     mu, sig = res.x
    
#     return mu, sig, res.fun

# def rt_rmse(df):
    
#     df = df.loc[(df.rt < 3000) & (df.rt > 400)]
#     rmse = np.sqrt(np.mean((df.s_rt - df.rt)**2))
    
#     return rmse

# def loftus_masson(df, sub_cols, cond_col, value_col, within_cols=[]):
    
#     if not isinstance(sub_cols, list):
#         sub_cols = [sub_cols]
#     if not isinstance(within_cols, list):
#         within_cols = [within_cols]
#     df = df.copy()
#     if len(within_cols) > 0:
#         df['M'] = df.groupby(within_cols)[value_col].transform('mean')
#     else:
#         df['M'] = df[value_col].mean()
#     df['M_S'] = df.groupby(sub_cols + within_cols)[value_col].transform('mean')
#     df['adj_' + value_col] = (df[value_col] + df['M'] - df['M_S'])
    
#     return df

# def rt_score(df):
    
#     df = df.loc[(df.rt < 3000) & (df.rt > 400) & (df.position > 10) & (df.old == df.yes)]
#     df = loftus_masson(df, 'subject_ID', [], 'rt')
    
#     a = 2800
#     c_thresh = 0.4
#     df['csim_diff'] = df['csim'] - c_thresh
#     df['csim_score'] = np.power(-1, df['yes']) * (np.log(df.adj_rt) - np.log(a)) 
#     df.csim_score = df.csim_score.astype("float")
#     mod = smf.ols(formula='csim_score ~ -1 + csim_diff', data=df)
#     res = mod.fit()
#     mse = res.mse_resid
    
#     return mse
    

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