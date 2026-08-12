from typing import Optional, Union, List, Dict
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from anndata import AnnData
from ..SEM import SEM, SEM1, SEM2, SEM3
from .._utils import cell_tri, cell_tri_dev
from scipy.sparse import csr_matrix
from scipy.stats import false_discovery_control
import scanpy as sc
# todo: pathway sum in mechanical_signal

def contact_signal(df_ligrec: pd.DataFrame,
                   sem: Optional[SEM] = None,
                   adata: Optional[AnnData] = None,
                   contact_key: Optional[str] = 'contacts',
                   lr_delimiter: str = '-',
                   heteromeric_delimiter: str = '_'):
    '''
    Contact signal inference

    signal matrix is stored as .obsp['ligand-receptor']

    signal names are stored as .uns['contact_signal_info']

    rows are sender cells, columns are receiver cells
    '''

    if df_ligrec.shape[0] == 0:
        raise ValueError("empty ligand-receptor DB")
    if sem is None: # sem is not provided, using adata contact matrix
        contact_matrix = adata.obsp[contact_key]
    else: # sem is provided
        if adata is None: # adata is not provided, use sem.adata
            adata = sem.adata
        else: # adata is provided, use input adata
            if adata is not sem.adata: # check if same adata
                Warning('Provide adata is not an attribute of sem, sem.adata will be unchanged')
        if sem.contact_matrix is None:
            print('compute cell-cell contact')
            sem.compute_contact()
        else:
            contact_matrix = sem.contact_matrix
    df_ligrec = df_ligrec.copy()
    # get cell pair index
    nc = adata.shape[0]
    # could be replace by ci,cj = contact_matrix.nonzero()
    indices = contact_matrix.indices
    indptr = contact_matrix.indptr
    ci = []
    cj = []
    for i in range(nc):
        j = indices[indptr[i]:indptr[i+1]]
        ci.append(np.tile(i,len(j)))
        cj.append(j)
    ci = np.concatenate(ci)
    cj = np.concatenate(cj)
    #
    # contact signal
    lr_keys = []
    I = np.ones(df_ligrec.shape[0],dtype=bool)
    # ligand-receptors pairs
    for i in range(df_ligrec.shape[0]):
        l = df_ligrec.iloc[i,0]
        r = df_ligrec.iloc[i,1]
        l_data = np.prod(adata[:,l.split(heteromeric_delimiter)].X[ci].toarray(),axis=1)
        r_data = np.prod(adata[:,r.split(heteromeric_delimiter)].X[cj].toarray(),axis=1)
        key = f'{l}{lr_delimiter}{r}'
        sig_mat = csr_matrix((l_data*r_data,indices.copy(),indptr.copy()), shape=(nc, nc)) # .copy() is necessary. eliminate_zeros() removes indices and indptr inplace
        sig_mat.eliminate_zeros()
        I[i] = sig_mat.nnz>0
        if I[i]:
            adata.obsp[f'contact_{key}'] = sig_mat
            lr_keys.append(key)
    df_ligrec = df_ligrec[I]

    # pathway and total
    pth_keys = df_ligrec.iloc[:,2].unique().tolist()
    for n,pth in enumerate(pth_keys):
        lr_idx = np.where(df_ligrec.iloc[:,2]==pth)[0]
        data = csr_matrix((nc,nc))
        for i in lr_idx:
            l = df_ligrec.iloc[i,0]
            r = df_ligrec.iloc[i,1]
            data += adata.obsp[f'contact_{l}{lr_delimiter}{r}'].copy()
        adata.obsp[f'contact_{pth}'] = data.copy()
        if n == 0:
            total = data.copy()
        else:
            total += data.copy()
    adata.obsp['contact_total'] = total
    adata.uns['contact_signal_info'] = {'signal': lr_keys, 'pathway': pth_keys, 'total': ['total'], 'db': df_ligrec}

    # receiver/sender signal
    signal_list = lr_keys + pth_keys + ['total']
    sdim = len(signal_list)
    signal_vec_s = np.zeros((adata.shape[0], sdim))
    signal_vec_r = np.zeros((adata.shape[0], sdim))
    for si,key in enumerate(signal_list):
        signal_vec_s[:,si] = adata.obsp[f'contact_{key}'].sum(axis=1).A1# sender signal
        signal_vec_r[:,si] = adata.obsp[f'contact_{key}'].sum(axis=0).A1# receiver signal
    s_col = [f's-{key}' for key in signal_list]
    r_col = [f'r-{key}' for key in signal_list]
    df_s = pd.DataFrame(index = adata.obs.index, columns=s_col,data=signal_vec_s)
    df_r = pd.DataFrame(index = adata.obs.index, columns=r_col,data=signal_vec_r)
    adata.obsm['sender_contact_signal'] = df_s
    adata.obsm['receiver_contact_signal'] = df_r
    print("add .obsm['sender_contact_signal'], .obsm['receiver_contact_signal'], .uns['contact_signal_info']")

def mechanical_signal(
        sem:Union[SEM, SEM1, SEM2, SEM3], 
        df_mechano:pd.DataFrame, 
        df_ligrec:pd.DataFrame,  
        heteromeric_delimiter:str = '_',
        lr_delimiter:str = '-',
        log1p_f: bool = True,
    ) -> None:
    """
    s[i,j]: signal j->i

    row index i: receiver

    col index j: sender
    """
    adata = sem.adata
    contact_matrix = sem.contact_matrix
    f_cc = sem.f_cc.copy()
    if log1p_f:
        print('apply log1p to cell-cell mechanical interaction')
        f_cc.data = np.log1p(f_cc.data)
    nc = sem.nc
    if df_mechano.shape[0] == 0:
        raise ValueError("empty DB")
    # ion channel, TF
    signal_list = []
    I = np.ones(df_mechano.shape[0],dtype=bool)
    for i,gene in enumerate(df_mechano['Symbol']):
        exp = adata[:,gene].X.toarray().flatten().reshape(-1, 1)
        sig_mat = f_cc.multiply(exp)
        sig_mat.setdiag(0)
        sig_mat.eliminate_zeros()
        I[i] = sig_mat.nnz>0
        if I[i]:
            sig_key = gene
            signal_list.append(sig_key)
            adata.obsp[f'mechano_{sig_key}'] = sig_mat.tocsr()
    df_mechano = df_mechano[I].copy()
    # contact lr 
    indices = contact_matrix.indices
    indptr = contact_matrix.indptr
    ci,cj = contact_matrix.nonzero()
    I = np.ones(df_ligrec.shape[0],dtype=bool)
    for i in range(df_ligrec.shape[0]):
        l = df_ligrec.iloc[i,0]
        r = df_ligrec.iloc[i,1]
        l_data = np.prod(adata[:,l.split(heteromeric_delimiter)].X[cj].toarray(),axis=1)# sender, j
        r_data = np.prod(adata[:,r.split(heteromeric_delimiter)].X[ci].toarray(),axis=1)# receiver, i
        # lr[i,j]: signal j-->i
        lr = csr_matrix((l_data*r_data,indices.copy(),indptr.copy()), shape=(nc, nc)) # .copy() is necessary. eliminate_zeros() removes indices and indptr inplace
        sig_mat = f_cc.multiply(lr)
        sig_mat.eliminate_zeros()
        I[i] = sig_mat.nnz>0
        if I[i]:
            sig_key = f'{l}{lr_delimiter}{r}'
            adata.obsp[f'mechano_{sig_key}'] = sig_mat
            # lr_keys.append(key)
            signal_list.append(sig_key)
    df_ligrec = df_ligrec[I].copy()
    # todo: pathway sum
    # pth_keys = df_ligrec.iloc[:,2].unique().tolist()
    # for n,pth in enumerate(pth_keys):
    #     lr_idx = np.where(df_ligrec.iloc[:,2]==pth)[0]
    #     data = csr_matrix((nc,nc))
    #     for i in lr_idx:
    #         l = df_ligrec.iloc[i,0]
    #         r = df_ligrec.iloc[i,1]
    #         data += adata.obsp[f'mechano_{l}{lr_delimiter}{r}']
    #     sig_key = pth
    #     adata.obsp[f'mechano_{sig_key}'] = data.copy()
        # signal_list.append(sig_key)
    
    # receiver/sender signal
    sdim = len(signal_list)
    signal_vec_s = np.zeros((adata.shape[0], sdim))
    signal_vec_r = np.zeros((adata.shape[0], sdim))
    for si, key in enumerate(signal_list):
        signal_vec_s[:,si] = adata.obsp[f'mechano_{key}'].sum(axis=0).A1# sender signal
        signal_vec_r[:,si] = adata.obsp[f'mechano_{key}'].sum(axis=1).A1# receiver signal
    s_col = [f's-{key}' for key in signal_list]
    r_col = [f'r-{key}' for key in signal_list]
    df_s = pd.DataFrame(index = adata.obs.index, columns=s_col,data=signal_vec_s)
    df_r = pd.DataFrame(index = adata.obs.index, columns=r_col,data=signal_vec_r)
    adata.obsm['sender_mechano_signal'] = df_s
    adata.obsm['receiver_mechano_signal'] = df_r
    adata.uns['mechano_signal_info'] = {'signal': signal_list, 'db': df_mechano, 'lr_db': df_ligrec}
    print("add .obsm['sender_mechano_signal'], .obsm['receiver_mechano_signal'], .uns['mechano_signal_info']")

def LR_signal(
        adata, 
        df_ligrec, 
        spatial_key = 'spatial', 
        heteromeric_delimiter = '_', 
        lr_delimiter = '-',
        key_added = 'LR',
):
    """
    s[i,j]: signal j->i

    row index i: receiver

    col index j: sender
    """
    
    xc = adata.obsm[spatial_key]
    nc = xc.shape[0]
    tri = Delaunay(xc)
    distance_matrix = lil_matrix((nc, nc))
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                d_ij = distance.euclidean(xc[simplex[i]],xc[simplex[j]])
                distance_matrix[simplex[i], simplex[j]] = d_ij
                distance_matrix[simplex[j], simplex[i]] = d_ij

    dc = np.zeros(nc)
    for cid in range(nc):
        _,j=distance_matrix[cid].nonzero()
        dc[cid] = np.mean(distance_matrix[cid,j]) if len(j)>0 else np.nan # some points might overlap with others
    I = np.isnan(dc)
    dc_m = np.median(dc[~I])
    params = {'dc_m':dc_m, 'gaussian_sigma': None}

    tree = KDTree(xc)
    r = dc_m*20
    pairs = tree.query_pairs(r, p=2, output_type='ndarray')
    # n_pairs = pairs.shape[0]
    i_indices = pairs[:, 0]
    j_indices = pairs[:, 1]

    dists = np.linalg.norm(xc[i_indices] - xc[j_indices], axis=1)
    row = np.concatenate([i_indices, j_indices])
    col = np.concatenate([j_indices, i_indices])
    data = np.concatenate([dists, dists])
    # if gaussian:
    sigma = r/4
    params['gaussian_sigma'] = sigma
    data = np.exp( -(data/sigma)**2/2 )
    dist_matrix = csr_matrix((data, (row, col)), shape=(nc, nc))

    # dist_matrix row normalize 
    row_sums = np.asarray(dist_matrix.sum(axis=1)).ravel()  # (nc,)
    inv = np.zeros_like(row_sums)
    nz = row_sums != 0
    inv[nz] = 1.0 / row_sums[nz]                             # 空行保持为 0，避免除零
    dist_matrix = diags(inv) @ dist_matrix                  # 每行和 = 1（空行仍为 0）

    # adata.obsp[f'{spatial_key}_distances'] = dist_matrix.copy()
    indices = dist_matrix.indices.copy()
    indptr = dist_matrix.indptr.copy()
    d = dist_matrix.data.copy()
    ci = []
    cj = []
    for i in range(nc):
        j = indices[indptr[i]:indptr[i+1]]
        ci.append(np.tile(i,len(j)))
        cj.append(j)
    ci = np.concatenate(ci)
    cj = np.concatenate(cj)

    lr_keys = []
    I = np.ones(df_ligrec.shape[0],dtype=bool)
    for i in tqdm(range(df_ligrec.shape[0])):
        l = df_ligrec.iloc[i,0]
        r = df_ligrec.iloc[i,1]
        l_data = np.prod(adata[:,l.split(heteromeric_delimiter)].X.toarray()[cj],axis=1) # sender, j
        r_data = np.prod(adata[:,r.split(heteromeric_delimiter)].X.toarray()[ci],axis=1) # receiver, i
        # l_data = np.sum(adata[:,l.split(heteromeric_delimiter)].X.toarray()[cj],axis=1) # sender, j
        # r_data = np.sum(adata[:,r.split(heteromeric_delimiter)].X.toarray()[ci],axis=1) # receiver, i
        lr_key = f'{key_added}_{l}{lr_delimiter}{r}'

        sig_mat = csr_matrix((l_data*r_data*d,indices.copy(),indptr.copy()), shape=(nc, nc))# .copy() is necessary. eliminate_zeros() removes indices and indptr inplace
        sig_mat.eliminate_zeros()
        I[i] = sig_mat.nnz>0
        if I[i]:
            adata.obsp[lr_key] = sig_mat
            lr_keys.append(lr_key)

    # df_ligrec_detected = df_ligrec[I]
    adata.uns['LR_signal_info'] = {'signal': lr_keys, 'db': df_ligrec, 'params': params}

def cluster_communication(adata: AnnData,
                          cluster_key: str,
                          signal: str = 'total',
                          prefix: str = 'mechano_',
                          n_permutations: int = 100,
                          seed: int = 0):
    """
    Cluster-cluster communication

    add cluster communication to .uns
    """
    
    cluster_list = list(adata.obs[cluster_key].cat.categories)
    cluster_cell = adata.obs[cluster_key].to_numpy()
    sig_mat = adata.obsp[prefix+signal]
    rng = np.random.default_rng(seed)
    tmp_df, tmp_p_value = summarize_cluster(sig_mat,cluster_cell,cluster_list,rng,n_permutations=n_permutations)
    key_add = cluster_key+'_'+signal
    if len(prefix)>0:
        key_add = cluster_key+'_'+prefix+signal 
    adata.uns[key_add] = {'communication_matrix': tmp_df, 'communication_pvalue': tmp_p_value}

# def signal_vector(adata: AnnData,
#                   signal_type: Optional[Union[List,str]] = ['lr_pair','pathway','total'],
#                   return_output=False):
#     """
#     Compute the sender signals and receiver signals for each cells

#     add 'sender_signal' 'receiver_signal' to .obsm
#     """
    
#     signal_type = [signal_type] if type(signal_type) is str else signal_type
#     signal_list = []
#     for key in signal_type:
#         signal_list+=adata.uns['contact_signal'][key]
#     sdim = len(signal_list)
#     signal_vec_s = np.zeros((adata.shape[0], sdim))
#     signal_vec_r = np.zeros((adata.shape[0], sdim))
#     for si,signal in enumerate(signal_list):
#         signal_vec_s[:,si] = np.sum(adata.obsp[signal].toarray(),axis=1)# sender signal
#         signal_vec_r[:,si] = np.sum(adata.obsp[signal].toarray(),axis=0)# receiver signal
#     df_s = pd.DataFrame(index = adata.obs.index, columns=signal_list,data=signal_vec_s)
#     df_r = pd.DataFrame(index = adata.obs.index, columns=signal_list,data=signal_vec_r)
#     adata.obsm['sender_signal'] = df_s
#     adata.obsm['receiver_signal'] = df_r
#     print("add 'sender_signal' 'receiver_signal' to .obsm")
#     if return_output:
#         return df_s, df_r

# def summarize_signal(adata: AnnData, cluster_key: str):
#     df_r = pd.DataFrame(index = adata.obs.index, columns=adata.uns['contact_signal'])
#     df_s = pd.DataFrame(index = adata.obs.index, columns=adata.uns['contact_signal'])
#     for sig_key in adata.uns['contact_signal']:
#         df_r[sig_key] = adata.obsp[sig_key].toarray().sum(axis=0)# receiver signal
#         df_s[sig_key] = adata.obsp[sig_key].toarray().sum(axis=1)# sender signal
#     selected_columns = [col for col in adata.uns['contact_signal'] if len(col.split('-')) >1]
#     df_r_sel = df_r[selected_columns]
#     df_s_sel = df_s[selected_columns]
#     df_r_sel['cell_type'] = adata.obs[cluster_key]
#     df_s_sel['cell_type'] = adata.obs[cluster_key]
#     return df_r_sel.groupby('cell_type').mean(),df_s_sel.groupby('cell_type').mean()
    
def summarize_cluster(X, clusterid, clusternames, rng, n_permutations):
    """
    X: CSR sparse matrix (n_cells x n_cells)
        - X[i,j]: signal j->i
        - row index i: receiver
        - col index j: sender
    
    clusterid: array of cluster labels (may contain NaN for unassigned cells)
    clusternames: list of valid cluster names
    
    核心加速：用矩阵乘法 M.T @ X @ M 替代 n^2 Python 循环 + sparse fancy indexing
    每次 permutation 从 O(n^2 * sparse_index_cost) -> O(nnz * n + n_cells * n^2)
    """
    n = len(clusternames)
    n_cells = X.shape[0]

    # Map cluster names to integers; NaN/unrecognized cells get -1
    name_to_int = {cn: i for i, cn in enumerate(clusternames)}
    cell_label = np.array([name_to_int.get(c, -1) for c in clusterid], dtype=np.int32)
    cluster_idx = [np.where(cell_label == k)[0] for k in range(n)]

    # Compute original cluster statistics (done once, Python loop OK)
    X_cluster = np.zeros((n, n), float)
    # X_th_cluster = np.zeros((n, n), float)
    for i in range(n):
        for j in range(n):
            block = X[cluster_idx[i], :][:, cluster_idx[j]]
            X_cluster[i, j] = block.mean()
            # npos = (block > 0).sum()
            # if npos > 0:
            #     X_th_cluster[i, j] = block.sum() / npos

    # Permutation test via matrix multiplication
    # For each permutation: build one-hot M (n_cells x n), then
    #   X_cluster_perm = (M.T @ X @ M) / outer(perm_sizes, perm_sizes)
    # This replaces the inner n^2 Python loop entirely.
    p_cluster = np.zeros((n, n), float)

    for _ in range(n_permutations):
        perm_labels = rng.permutation(cell_label)  # shuffles valid labels AND -1 (NaN) cells

        # Build membership matrix: M[c, k] = 1 iff cell c assigned to cluster k
        valid = perm_labels >= 0
        M = np.zeros((n_cells, n), dtype=np.float32)
        M[np.where(valid)[0], perm_labels[valid]] = 1.0

        # Per-permutation cluster sizes (vary because NaN cells swap with valid cells)
        perm_sizes = M.sum(axis=0)  # shape (n,)
        size_prod = np.outer(perm_sizes, perm_sizes)
        size_prod[size_prod == 0] = 1.0  # avoid division by zero for empty clusters

        # sparse @ dense: (n_cells, n_cells) @ (n_cells, n) -> (n_cells, n)
        XM = X @ M
        # (n, n_cells) @ (n_cells, n) -> (n, n)
        X_cluster_perm = (M.T @ XM) / size_prod

        p_cluster += (X_cluster_perm >= X_cluster)

    p_cluster = (p_cluster + 1) / (n_permutations + 1)
    q_cluster = false_discovery_control(p_cluster.flatten()).reshape([n, n])

    df_X = pd.DataFrame(data=X_cluster, index=clusternames, columns=clusternames)
    df_pvalue = pd.DataFrame(data=p_cluster, index=clusternames, columns=clusternames)
    df_qvalue = pd.DataFrame(q_cluster, index=clusternames, columns=clusternames)
    return df_X, df_pvalue, df_qvalue

def cell2grid(xc, Nx, padding = 1e-3):
    # padding>0 must
    xc_min = xc.min(axis=0)
    xc_max = xc.max(axis=0)
    xc_L = xc_max - xc_min
    xc_m = (xc_max+xc_min)/2
    a = 1+padding
    
    grid_step = xc_L[0]/Nx*a
    Ny = np.ceil(xc_L[1]/xc_L[0]*Nx/a).astype(int)
    x_grid = np.linspace(0,xc_L[0]*a,Nx+1) - (xc_L[0]*a/2-xc_m[0])
    y_grid = np.arange(Ny+1)*grid_step- (xc_L[1]*a/2-xc_m[1])

    if not np.all((xc[:,0]>x_grid[0]) & (xc[:,0]<x_grid[-1]) & (xc[:,1]>y_grid[0]) & (xc[:,1]<y_grid[-1])):
        print('cells outside grid') # warning

    xc2grid_map = dict()
    grid_center = dict()
    for i in range(Nx):
        Ix = (xc[:,0]>x_grid[i]) & (xc[:,0]<x_grid[i+1])
        for j in range(Ny):
            Iy = (xc[:,1]>y_grid[j]) & (xc[:,1]<y_grid[j+1])
            idx = np.where(Ix&Iy)[0]
            if idx.shape[0]>0:
                xc2grid_map[(i,j)] = idx
                grid_center[(i,j)] = xc[idx].mean(axis=0) # mean cell location
                # grid_center[(i,j)] = np.array([X_grid[j,i],Y_grid[j,i]])

    return xc2grid_map, grid_center, x_grid, y_grid

def signal_direction(xc, sig_mat, xc2grid_map = None, Nc_grid_min = 3, W_grid_min = 0.):
    indices = sig_mat.indices.copy()
    indptr = sig_mat.indptr.copy()
    data = sig_mat.data.copy()
    nc,dim = xc.shape
    Vc = np.zeros((nc,dim))
    Sc = sig_mat.sum(axis=1).A1
    for cid_i in range(nc):
        v = np.zeros((1,dim))
        # sig_sum = 0
        for ptr in range(indptr[cid_i],indptr[cid_i+1]):
            cid_j = indices[ptr]
            v += (xc[cid_i] - xc[cid_j])*data[ptr]
            # sig_sum += data[ptr]
        if Sc[cid_i]>0:
            Vc[cid_i] = v/Sc[cid_i]
    
    if xc2grid_map is None:
        return Vc
    
    # to grid
    # if xc2grid_map is not None:
        # V2grid_map = dict()
    Wgrid = []
    Vgrid = []
    Xgrid = []
    for ij in xc2grid_map:
        cid = xc2grid_map[ij]
        # V2grid_map[ij] = Vc[cid].mean(axis=0)
        w = Sc[cid]
        Wgrid.append(w.sum())
        if w.sum()>0:
            #\bar{V}{ij} = \frac{\sum{c \in \text{cid}} S_c \cdot V_c}{\sum_{c \in \text{cid}} S_c}
            Vgrid.append(np.average(Vc[cid], axis=0, weights=w))
            Xgrid.append(np.average(xc[cid], axis=0, weights=w))
        else:
            Vgrid.append(np.zeros(dim))
            Xgrid.append(xc[cid].mean(axis=0))
    Wgrid = np.array(Wgrid)
    Vgrid = np.array(Vgrid)
    Xgrid = np.array(Xgrid)#list(grid_center.values())
    Nc_grid = np.array( [len(xc2grid_map[ij]) for ij in xc2grid_map] )
    I_grid = (Nc_grid>=Nc_grid_min) & (Wgrid>W_grid_min)

    return Xgrid[I_grid], Vgrid[I_grid], Wgrid[I_grid], Vc

def signal_tensor(xc, xc2grid_map, grid_center, sig_mat, Nc_grid_min = 3):
    indices = sig_mat.indices.copy()
    indptr = sig_mat.indptr.copy()
    data = sig_mat.data.copy()
    nc,dim = xc.shape

    Sc = np.zeros((nc,dim,dim))
    for cid_i in range(nc):
        s = np.zeros((dim,dim))
        # sig_sum = 0
        for ptr in range(indptr[cid_i],indptr[cid_i+1]):
            cid_j = indices[ptr]
            # v = xc[cid_i] - xc[cid_j]
            v = (xc[cid_j] - xc[cid_i])/np.linalg.norm(xc[cid_j] - xc[cid_i])
            s += np.outer(v,v)*data[ptr]
            # sig_sum+=data[ptr]
        Sc[cid_i] = s

    Ve2grid_map = dict()
    Sgrid = []
    for ij in xc2grid_map:
        cid = xc2grid_map[ij]
        Sg = Sc[cid].mean(axis=0)
        Sgrid.append( Sg )
        eig_val, eig_vec = np.linalg.eig( Sg ) # column eig_vec[:,i] is the eigenvector corresponding to the eigenvalue eig_val[i].
        sort_i = np.argsort(eig_val)
        Ve2grid_map[ij] = (eig_vec*eig_val)[:,sort_i]
        # Ve2grid_map[ij] = (eig_vec.T*eig_val)[sort_i].T

    Nc_grid = np.array( [len(xc2grid_map[ij]) for ij in xc2grid_map] )
    I_grid = Nc_grid>=Nc_grid_min
    Sgrid = np.array(Sgrid)
    Xgrid = np.array(list(grid_center.values()))
    Vegrid = np.array(list(Ve2grid_map.values()))
    return Xgrid[I_grid], Vegrid[I_grid], Sgrid[I_grid]

def signal_enrichment(
        sem: Union[SEM, SEM1, SEM2, SEM3], 
        groupby: str, 
        obsm_key: str = 'receiver_mechano_signal',
        spatial_key: Optional[str] = None
):
    
    mat = sem.adata.obsm[obsm_key]
    var_df = pd.DataFrame(index=mat.columns)
    obs_df = sem.adata.obs.copy()
    adata_signal = AnnData(X=mat,obs=obs_df,var=var_df)
    group_counts = obs_df[groupby].value_counts()
    small_groups = group_counts[group_counts == 1].index.tolist()# 2. 找出只含 1 个细胞的分组
    I = ~obs_df[groupby].isin(small_groups)# 3. 过滤掉这些分组对应的细胞
    adata_signal_copy = adata_signal[I].copy()
    sc.tl.rank_genes_groups(adata_signal_copy,groupby=groupby,method='wilcoxon',pts=True)
    adata_signal.uns['rank_genes_groups'] = adata_signal_copy.uns['rank_genes_groups']
    adata_signal.obs['selected'] = I
    if spatial_key is not None:
        adata_signal.obsm[spatial_key] = sem.adata.obsm[spatial_key].copy()
    sem.adata_signal = adata_signal
    print('add .adata_signal')
    # todo: dict to store adata_signal for 'receiver_mechano_signal', 'sender_mechano_signal'
    # todo: concate 'receiver_mechano_signal', 'sender_mechano_signal'

def contact_analysis(sem: Union[SEM1, SEM2]):
    ctri = cell_tri(sem)
    print(f'create ctri for sem {sem.sim_name}')
    ctri.compute_shape()
    print('add ctri.cell_boundary')
    # ctri.compute_contact()
    # print('add ctri.contact_matrix')
    return ctri

def contact_analysis_test(sem: SEM2, cid_list: Optional[NDArray[np.int_]] = None):
    """test for SEM2 and cell_tri_dev"""
    ctri = cell_tri_dev(sem, cid_list)
    print(f'create ctri for sem {sem.sim_name}')
    ctri.compute_shape()
    print('add ctri.cell_boundary')
    # ctri.compute_contact()
    # print('add ctri.contact_matrix')
    return ctri

def pressure_analysis(sem: SEM1, key_add:str = 'pressure'):
    if not hasattr(sem,'p'):
        sem.compute_pressure()
    p_c = np.zeros(sem.nc)
    for i in range(sem.nc):
        p_c[i] = np.mean(sem.p[sem.ceidn[i]:sem.ceidn[i+1]])
    if sem.adata is not None:
        sem.adata.obs[key_add] = p_c
        print(f"add .obs[{key_add}]")
    return p_c