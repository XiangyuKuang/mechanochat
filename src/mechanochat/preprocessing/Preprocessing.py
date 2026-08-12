import numpy as np
import pandas as pd
from pandas import DataFrame
from anndata import AnnData
import pkgutil
import io
from .._utils import scale
from sklearn.preprocessing import minmax_scale
from typing import Optional, Union, Tuple, Dict, List, Iterable, Literal
from numpy.typing import NDArray
import warnings

def ligand_receptor_database(
    database: Literal["CellChat","CellPhoneDB"] = "CellChat",
    species: Literal["mouse","human"] = "mouse",
    signaling_type: Literal["Secreted Signaling","Cell-Cell contact","ECM-Receptor"] = "Secreted Signaling"
) -> DataFrame:
    """
    Extract ligand-receptor pairs from LR database.

    Parameters
    ----------
    database
        The name of the ligand-receptor database. Use 'CellChat' for CellChatDB [Jin2021]_ of 'CellPhoneDB' for CellPhoneDB_v4.0 [Efremova2020]_.
    species
        The species of the ligand-receptor pairs. Choose between 'mouse' and 'human'.
    heteromeric_delimiter
        The character to separate the heteromeric units of heteromeric ligands and receptors. 
        For example, if the heteromeric receptor (TGFbR1, TGFbR2) will be represented as 'TGFbR1_TGFbR2' if this parameter is set to '_'.
    signaling_type
        The type of signaling. Choose from 'Secreted Signaling', 'Cell-Cell Contact', and 'ECM-Receptor' for CellChatDB or 'Secreted Signaling' and 'Cell-Cell Contact' for CellPhoneDB_v4.0. 
        If None, all pairs in the database are returned.

    Returns
    -------
    df_ligrec : pandas.DataFrame
        A pandas DataFrame of the LR pairs with the three columns representing the ligand, receptor, and the signaling pathway name, respectively.

    References
    ----------

    .. [Jin2021] Jin, S., Guerrero-Juarez, C. F., Zhang, L., Chang, I., Ramos, R., Kuan, C. H., ... & Nie, Q. (2021). 
        Inference and analysis of cell-cell communication using CellChat. Nature communications, 12(1), 1-20.
    .. [Efremova2020] Efremova, M., Vento-Tormo, M., Teichmann, S. A., & Vento-Tormo, R. (2020). 
        CellPhoneDB: inferring cell–cell communication from combined expression of multi-subunit ligand–receptor complexes. Nature protocols, 15(4), 1484-1506.

    """
    assert database in ("CellChat", "CellPhoneDB")
    assert species in ("mouse", "human", "zebrafish")
    assert signaling_type in ("Secreted Signaling", "Cell-Cell Contact", "ECM-Receptor")
    if database == "CellChat":
        data = pkgutil.get_data(__name__, "_data/LRdatabase/CellChatDB.ligrec."+species+".csv")
        df_ligrec = pd.read_csv(io.BytesIO(data), index_col=0)
        if not signaling_type is None:
            df_ligrec = df_ligrec[df_ligrec.iloc[:,3] == signaling_type]
    elif database == 'CellPhoneDB':
        data = pkgutil.get_data(__name__, "_data/LRdatabase/CellPhoneDBv4.0."+species+".csv")
        df_ligrec = pd.read_csv(io.BytesIO(data), index_col=0)
        if not signaling_type is None:
            df_ligrec = df_ligrec[df_ligrec.iloc[:,3] == signaling_type]
    df_ligrec.columns = ['Ligand', 'Receptor', 'Pathway', 'Type']
    return df_ligrec

def filter_lr_database(
    df_ligrec: DataFrame,
    adata: AnnData,
    heteromeric: bool = True,
    heteromeric_delimiter: str = "_",
    heteromeric_rule: Literal["min","ave"] = "min",
    filter_criteria: Literal["min_cell_pct","min_cell"] = "min_cell_pct",
    min_cell: int = 100,
    min_cell_pct: float = 0.05
) -> DataFrame:
    """
    Filter ligand-receptor pairs.

    Parameters
    ----------
    df_ligrec
        The pandas dataframe of ligand-receptor database with three columns being ligand, receptor, and pathway name respectively.
    adata
        The AnnData object of gene expression. Unscaled data (minimum being zero) is expected.
    heteromeric
        Whether the ligands and receptors are described as heteromeric.
    heteromeric_delimiter
        If heteromeric notations are used for ligands and receptors, the character separating the heteromeric units.
    heteromeric_rule
        When  heteromeric is True, the rule to quantify the level of a heteromeric ligand or receptor. Choose from minimum ('min') and average ('ave').
    filter_criteria
        Use either cell percentage ('min_cell_pct') or cell numbers (min_cell) to filter genes.
    min_cell
        If filter_criteria is 'min_cell', the LR-pairs with both ligand and receptor detected in greater than or equal to min_cell cells are kept.
    min_cell_pct
        If filter_criteria is 'min_cell_pct', the LR-pairs with both ligand and receptor detected in greater than or equal to min_cell_pct percentage of cells are kepts.

    Returns
    -------
    df_ligrec_filtered: pd.DataFrame
        A pandas DataFrame of the filtered ligand-receptor pairs.

    """
    data_genes = set(adata.var_names)
    ncell = adata.shape[0]
    all_genes = list(adata.var_names)
    gene_ncell = np.array( (adata.X > 0).sum(axis=0) ).reshape(-1)
    ligrec_list = []
    genes_keep = []
    if not heteromeric:
        tmp_genes = set(df_ligrec.iloc[:,0]).union(set(df_ligrec.iloc[:,1]))
        tmp_genes = list(tmp_genes.intersection(data_genes))
        for gene in tmp_genes:
            if not gene in all_genes: continue
            if filter_criteria == 'min_cell_pct':
                if gene_ncell[all_genes.index(gene)] / ncell >= min_cell_pct:
                    genes_keep.append(gene)
            elif filter_criteria == 'min_cell':
                if gene_ncell[all_genes.index(gene)] >= min_cell:
                    genes_keep.append(gene)
    elif heteromeric:
        tmp_genes = list(set(df_ligrec.iloc[:,0]).union(set(df_ligrec.iloc[:,1])))
        for het_gene in tmp_genes:
            genes = het_gene.split(heteromeric_delimiter)
            gene_found = True
            for gene in genes:
                if not gene in all_genes:
                    gene_found = False
            if not gene_found: continue
            keep = True
            if filter_criteria == 'min_cell_pct' and heteromeric_rule == 'min':
                for gene in genes:
                    if gene_ncell[all_genes.index(gene)] / ncell < min_cell_pct:
                        keep = False
            elif filter_criteria == 'min_cell' and heteromeric_rule == 'min':
                for gene in genes:
                    if gene_ncell[all_genes.index(gene)] < min_cell:
                        keep = False
            elif heteromeric_rule == 'ave':
                ave_ncell = []
                for gene in genes:
                    ave_ncell.append( gene_ncell[all_genes.index(gene)] )
                if filter_criteria == 'min_cell_pct':
                    if np.mean(ave_ncell) / ncell < min_cell_pct:
                        keep = False
                elif filter_criteria == 'min_cell':
                    if np.mean(ave_ncell) < min_cell:
                        keep = False
            if keep:
                genes_keep.append(het_gene)
    for i in range(df_ligrec.shape[0]):
        if df_ligrec.iloc[i,0] in genes_keep and df_ligrec.iloc[i,1] in genes_keep:
            ligrec_list.append(list(df_ligrec.iloc[i,:]))
    
    return DataFrame(data=ligrec_list, columns = ['Ligand', 'Receptor', 'Pathway', 'Type'])

def load_mechano_db(
        species: Literal["mouse", "human"]
) -> Tuple[DataFrame,DataFrame,DataFrame,DataFrame]:
    
    assert species in ("mouse", "human")
    # stiffness
    data1 = pkgutil.get_data(__name__, f"_data/MCdatabase/Stiffness_{species}.csv")
    df_stiffness = pd.read_csv(io.BytesIO(data1), index_col=0)
    # mechano-sensing ion channel, TF
    data2 = pkgutil.get_data(__name__, f"_data/MCdatabase/MechanoSensing_{species}.csv")
    df_IonTF = pd.read_csv(io.BytesIO(data2), index_col=None)

    df_cellchat = ligand_receptor_database("CellChat",species,"Cell-Cell Contact")
    # adhesion
    adh_pth = ['CADM','CDH','CDH1','CDH5','ICAM','JAM','NCAM','VCAM','ITGAL-ITGB2','CNTN','SELPLG','DESMOSOME']
    I = df_cellchat.iloc[:,2].isin(adh_pth)
    df_adh = df_cellchat[I].copy()
    df_adh["sorted_selected"] = df_adh[['Ligand','Receptor']].apply(lambda x: tuple(sorted(x)), axis=1)
    df_adh = df_adh.drop_duplicates(subset="sorted_selected").drop(columns="sorted_selected").reset_index(drop=True)
    # mechano-sensing lr
    lr_pth = ['Itg','Notch']
    if species == 'human':
        lr_pth = [x.upper() for x in lr_pth]
    I1 = np.zeros((df_cellchat.shape[0],len(lr_pth)))
    for n,t in enumerate(lr_pth):
        I1[:,n] = df_cellchat.iloc[:,1].str.contains(t)
    df_lr = df_cellchat[np.any(I1,axis=1)].copy()
    Dlk = 'Dlk' if species == "mouse" else 'DLK'
    df_lr = df_lr[~df_lr.iloc[:,0].str.contains(Dlk)].reset_index(drop=True)
    # I_itg = df_lr.iloc[:,1].str.contains(lr_pth[0], na=False)
    # df_lr.iloc[I_itg, 2] = 'Integrin'
    return df_stiffness, df_adh, df_IonTF, df_lr

def filter_db(
        df: DataFrame,
        adata: AnnData,
        filter_criteria: Literal["min_cell_pct", "min_cell"] = "min_cell_pct",
        min_cell: int = 100,
        min_cell_pct: float = 0.05
) -> DataFrame:
    
    assert filter_criteria in ("min_cell_pct", "min_cell")
    tmp_genes = df.iloc[:,0]
    all_genes = list(adata.var_names)
    genes_keep = []
    I = np.zeros(df.shape[0],dtype=bool)
    ncell = adata.shape[0]
    gene_ncell = np.array( (adata.X > 0).sum(axis=0) ).reshape(-1)
    for n,gene in enumerate(tmp_genes):
        if not gene in all_genes: continue
        if filter_criteria == 'min_cell_pct':
            if gene_ncell[all_genes.index(gene)] / ncell >= min_cell_pct:
                genes_keep.append(gene)
                I[n] = True
        elif filter_criteria == 'min_cell':
            if gene_ncell[all_genes.index(gene)] >= min_cell:
                genes_keep.append(gene)
                I[n] = True
    return df.iloc[I,:].reset_index(drop=True)

def get_params(
        adata: AnnData, 
        df_adh: DataFrame, 
        df_stiffness: DataFrame,
        adh_range: Tuple[float, float] = (0.5,3.5), 
        stiffness_range: Tuple[float, float] = (0.5,2.5),
) -> Tuple[NDArray, NDArray]:
    
    nc = adata.shape[0]
    lr = np.zeros((nc,nc))
    for i in range(df_adh.shape[0]):
        l = df_adh.iloc[i,0]
        r = df_adh.iloc[i,1]
        l_temp = adata[:,l.split('_')].X.toarray()
        r_temp = adata[:,r.split('_')].X.toarray()
        l_data = np.prod(l_temp-l_temp.min(axis=0),axis=1)
        r_data = np.prod(r_temp-r_temp.min(axis=0),axis=1)
        lr += np.matmul(r_data[np.newaxis].T,l_data[np.newaxis])+np.matmul(l_data[np.newaxis].T,r_data[np.newaxis])
    lr-=lr*np.eye(nc,nc)
    # high_adh = np.percentile(lr.mean(axis=0),95)
    # adhesion = np.minimum(lr,high_adh)
    # adhesion = minmax_scale(adhesion,adh_range) # debug: replace minmax_scale()

    mask = ~np.eye(nc, dtype=bool)
    high_adh = np.percentile(lr[mask],95)
    adhesion = np.minimum(lr,high_adh)
    adhesion,_,_ = scale(adhesion,adh_range,mask)

    stiffness_exp = adata[:,df_stiffness.iloc[:,0]].X.toarray()
    stiffness_exp_m = stiffness_exp.mean(axis=1)
    high_stiffness = np.percentile(stiffness_exp_m,95)
    stiffness = np.minimum(stiffness_exp_m,high_stiffness)
    stiffness,_,_ = scale(stiffness,stiffness_range)
    # stiffness = minmax_scale(stiffness,stiffness_range)
    
    return adhesion, stiffness

def signal_adata(
        adata: AnnData,
        obsm_key: str = 'receiver_mechano_signal',
        spatial_key: str = 'spatial',
) -> AnnData:
    
    mat = adata.obsm[obsm_key]
    obs_df = adata.obs.copy()
    adata_signal = AnnData(X=mat,obs=obs_df)
    if spatial_key is not None:
        adata_signal.obsm[spatial_key] = adata.obsm[spatial_key].copy()
    # sc.pp.filter_cells(adata_signal,min_genes=1)
    # sc.pp.filter_genes(adata_signal,min_cells=100)
    return adata_signal