import scipy as sp

def get_roc_df(df, grouping_cols = [], confidence_col='confidence_cat', 
               subject_col='subject_ID', old_col='old', hack_z=False):
    roc_df = df.groupby(grouping_cols + [subject_col] + [confidence_col] + [old_col]).size().to_frame(
        name='count').reset_index()
    roc_df.loc[roc_df['count'] == 0, 'count'] = .5
    roc_df['cumsum'] = roc_df.groupby(grouping_cols + [subject_col] + [old_col])['count'].cumsum()
    roc_df['total'] = roc_df.groupby(grouping_cols + [subject_col] + [old_col])['count'].transform('sum')
#     if hack_z:
#         roc_df['cumsum'] += .5
#         roc_df['total'] += 4
    roc_df['rate'] = 1 - roc_df['cumsum'] / roc_df['total']
    roc_df['z_rate'] = sp.stats.norm.ppf(roc_df['rate'])
    
    agg_roc_df = roc_df.groupby(grouping_cols + [confidence_col] + [old_col]).agg(
        {'rate': ['mean', 'sem']}).pivot_table(index=grouping_cols + [confidence_col], columns=old_col)
    agg_roc_df.columns = ['far', 'hr', 'far_sem', 'hr_sem']
    agg_roc_df['far_err'] = agg_roc_df['far_sem'] * 1.96
    agg_roc_df['hr_err'] = agg_roc_df['hr_sem'] * 1.96
    agg_roc_df = agg_roc_df.reset_index()
    
    agg_zroc_df = roc_df.groupby(grouping_cols + [confidence_col] + [old_col]).agg(
        {'z_rate': ['mean', 'sem']}).pivot_table(index=grouping_cols + [confidence_col], columns=old_col)
    agg_zroc_df.columns = ['z_far', 'z_hr', 'z_far_sem', 'z_hr_sem']
    agg_zroc_df['z_far_err'] = agg_zroc_df['z_far_sem'] * 1.96
    agg_zroc_df['z_hr_err'] = agg_zroc_df['z_hr_sem'] * 1.96
    agg_zroc_df = agg_zroc_df.reset_index()
    return agg_roc_df, agg_zroc_df, roc_df


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

def loglag_rollcat(df, resp_col):

    df_rollcat_laggp = df.groupby(['subject_ID','roll_cat_len_level','log_lag_bin'])[resp_col].mean()
    df_rollcat_laggp = df_rollcat_laggp.to_frame(name='hr').reset_index()
    df_rollcat_laggp = au.loftus_masson(df_rollcat_laggp, 'subject_ID', ['roll_cat_len_level', 'log_lag_bin'], 'hr')
    
    df_rollcat_laggp = df_rollcat_laggp.loc[np.isin(df_rollcat_laggp.roll_cat_len_level,['0','>=2'])]
    df_rollcat_laggp.roll_cat_len_level = df_rollcat_laggp.roll_cat_len_level.astype("category").cat.reorder_categories(['0', '>=2'])
    
    df_rollcat_laggp['log_lag_disp'] = np.e**df_rollcat_laggp.log_lag_bin
    
    display(df_rollcat_laggp)
    
    if resp_col == 's_resp':
        ci = None
        linestyle = '-'
    else:
        ci = 95
        linestyle = '--'

    g=sns.lineplot(data=df_rollcat_laggp, y='adj_hr', x='log_lag_disp', hue = 'roll_cat_len_level', ci=ci, linestyle='--', marker = 'o')
    g.set(ylabel='P("Yes" | Old)', xlabel='log_Lag')
    
    selected_lag = np.array([1,np.e,np.e**2,np.e**3,np.e**4,np.e**5])
    plt.xticks(ticks=selected_lag, labels = ['1','e','e2','e3','e4','e5'])
    plt.xlim([0,150])