import numpy as np
import pandas as pd
import xgboost as xgb
import sys
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

def XGBmodel_single(cur_row,exp):

    cur_TF = cur_row.iloc[0,1]
    cur_TG = cur_row.iloc[0,2]
    Recs_target = cur_row['Receptor'].tolist()
    exp_TF = np.array(exp[cur_TF])
    exp_TG = np.array(exp[cur_TG])

    exp_TFTG = exp_TF*exp_TG
    # exp_TFTG = np.log2(1+exp_TFTG)
    exp_recs = exp[Recs_target]
    # exp_recs = np.log2(1+exp_recs)

    xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=5, random_state=2024,device = 'cuda')
    xgb_model.fit(exp_recs,exp_TFTG)

    importance = xgb_model.get_booster().get_score(importance_type='gain')
    df_imp = pd.DataFrame(list(importance.items()), columns=['Receptor', 'importance'])
    df_imp['TF'] = cur_TF
    df_imp['TG'] = cur_TG
    df_imp = df_imp[['Receptor','TF','TG','importance']]

    return df_imp

def RFmodel_single(cur_row,exp):

    cur_TF = cur_row.iloc[0,1]
    cur_TG = cur_row.iloc[0,2]
    Recs_target = cur_row['Receptor'].tolist()
    exp_TF = np.array(exp[cur_TF])
    exp_TG = np.array(exp[cur_TG])

    exp_TFTG = exp_TF*exp_TG
    # exp_TFTG = np.log2(1+exp_TFTG)
    exp_recs = exp[Recs_target]
    # exp_recs = np.log2(1+exp_recs)

    rf_model = RandomForestRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=2024,device = 'cuda')
    rf_model.fit(exp_recs,exp_TFTG)

    importance = permutation_importance(rf_model, exp_recs,exp_TFTG, n_repeats=20,random_state=0)

    df_imp = pd.DataFrame(data = {'Receptor': Recs_target, 'importance': importance.importances_mean})
    df_imp['TF'] = cur_TF
    df_imp['TG'] = cur_TG
    df_imp = df_imp[['Receptor','TF','TG','importance']]

    return df_imp


def Regressor(df, exp_nor, method = "xgboost"):
    # method: xgboost, Random forest
    # input 1: dataframe of activated pathways: Receptor, TF, TG (as 3 columns)
    # input 2: normalized expression matrix 
    # ouput: dataframe of importance score: Receptor, TF, TG, Strength (as 4 columns)
    
    # Ensure the DataFrame has a unique index
    df = df.reset_index(drop=True)

    # Grouping DataFrame by 'y' and 'z'
    grouped = df.groupby(['TF', 'TG'])
    
    print("Start regression ...")
    
    # with mp.Pool(24) as pool:
    #     results = pool.starmap(RFmodel_single, tasks)
    count = 0
    res_list = [None for _ in range(len(grouped))]
    with tqdm(total=len(grouped)) as pbar:
        pbar.set_description('Processing:')
        for _,group in grouped:
            imp = XGBmodel_single(group,exp_nor)
            res_list[count] = imp
            count+=1
            pbar.update(1)

    # Flatten the list of results and sort by the original index
    flat_results = pd.concat(res_list)

    return flat_results