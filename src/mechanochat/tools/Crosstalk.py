from .._utils.SigXTalkpy import predictor as pred
from .._utils.SigXTalkpy import training as tr
from .._utils.SigXTalkpy import preprocessing as pp
from .._utils.SigXTalkpy import regression_model as rm

import torch
import numpy as np
import random

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def crosstalk_analysis(args, input_exp, RecTFDB, TFTGDB):
    
    ## PART 0: hyperparameters
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed)

    # print(f"Dataset: {args.project}, target type: {args.target_type}")

    ## PART 1: Load the input expression matrix & databases
    gene_all = input_exp.columns.tolist()
    # gene_all = input_exp.index.tolist()

    ## PART 2: Construct the hypergraph model and generate the training samples
    df_rt = pp.calculate_corr(input_exp, RecTFDB, type = args.corr_type, abss = False)
    df_tftg = pp.calculate_corr(input_exp, TFTGDB, type = args.corr_type, abss = False)
    df_rt = df_rt[~df_rt['Correlation'].isna()]
    
    hg, all_samples, input_all = pp.Filter_RTTDB(
        df_rt, 
        df_tftg, 
        thres1 = args.thres[0], 
        thres2 = args.thres[1], 
        genes = gene_all,
        args = args, 
        first = 'score', 
        sample_scale = 0.75, 
        ood_frac = 0.5
    )

    mymodel = pred.HGNNPredictor(
        in_channels = input_exp.shape[0],
        hgnn_channels = args.hgnn_dims,
        linear_channels = args.linear_dims
    )

    ## PART 3: Train the HGNN
    print(f"start training...")
    mymodel = tr.Train(args, input_exp.T, mymodel, hg, all_samples, device)

    pred_results = tr.Predict(Exp = input_exp.T, model = mymodel, input_samples = input_all, hypergraph = hg, genes = gene_all, device = device)
    
    # PART 4: Perform the regression to obtain PRS
    pred_filtered = pred_results.loc[pred_results['pred_label'] >= 0.75,:]
    pred_filtered = pred_filtered.reset_index()
    df = pred_filtered[['Receptor',"TF","TG"]]

    importances = rm.Regressor(df, input_exp)
    importances['fidelity'] = importances['importance']/importances.groupby('TG')['importance'].transform('sum')
    importances['specificity'] = importances['importance']/importances.groupby('Receptor')['importance'].transform('sum')

    return pred_results, importances