import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from scipy.spatial import KDTree, Delaunay, distance
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix, diags
import scanpy as sc
import pandas as pd
from tqdm import tqdm

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
        
# def signal_direction2(xc, sig_mat, xc2grid_map = None, Nc_grid_min = 3, W_grid_min = 0.):
#     # not finished
#     indices = sig_mat.indices.copy()
#     indptr = sig_mat.indptr.copy()
#     data = sig_mat.data.copy()
#     nc,dim = xc.shape
#     Vc = np.zeros((nc,dim))
#     Sc = sig_mat.sum(axis=1).A1
#     for cid_i in range(nc):
#         v = np.zeros((1,dim))
#         # sig_sum = 0
#         for ptr in range(indptr[cid_i],indptr[cid_i+1]):
#             cid_j = indices[ptr]
#             v_ij = (xc[cid_i] - xc[cid_j])
#             v += v_ij/np.linalg.norm(v_ij)*data[ptr]
#             # sig_sum += data[ptr]
#         # if Sc[cid_i]>0:
#             Vc[cid_i] = np.linalg.norm(v)
    
#     if xc2grid_map is None:
#         return Vc
    
#     # to grid
#     # if xc2grid_map is not None:
#         # V2grid_map = dict()
#     Wgrid = []
#     Vgrid = []
#     Xgrid = []
#     for ij in xc2grid_map:
#         cid = xc2grid_map[ij]
#         # V2grid_map[ij] = Vc[cid].mean(axis=0)
#         w = Sc[cid]
#         Wgrid.append(w.sum())
#         if w.sum()>0:
#             #\bar{V}{ij} = \frac{\sum{c \in \text{cid}} S_c \cdot V_c}{\sum_{c \in \text{cid}} S_c}
#             Vgrid.append(np.average(Vc[cid], axis=0, weights=w))
#             Xgrid.append(np.average(xc[cid], axis=0, weights=w))
#         else:
#             Vgrid.append(np.zeros(dim))
#             Xgrid.append(xc[cid].mean(axis=0))
#     Wgrid = np.array(Wgrid)
#     Vgrid = np.array(Vgrid)
#     Xgrid = np.array(Xgrid)#list(grid_center.values())
#     Nc_grid = np.array( [len(xc2grid_map[ij]) for ij in xc2grid_map] )
#     I_grid = (Nc_grid>=Nc_grid_min) & (Wgrid>W_grid_min)

#     return Xgrid[I_grid], Vgrid[I_grid], Wgrid[I_grid], Vc

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

def LR_signal(
        adata, 
        df_ligrec, 
        spatial_key = 'spatial', 
        heteromeric_delimiter = '_', 
        lr_delimiter = '-',
        key_added = 'LR',
        gaussian = True
):
    """
    s[i,j]: signal j->i

    row index i: receiver

    col index j: sender

    no dist_matrix normalize 
    """
    
    xc = adata.obsm[spatial_key]
    nc = xc.shape[0]
    tri = Delaunay(xc)
    distance_matrix = lil_matrix((nc, nc))
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                d = distance.euclidean(xc[simplex[i]],xc[simplex[j]])
                distance_matrix[simplex[i], simplex[j]] = d
                distance_matrix[simplex[j], simplex[i]] = d

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
    if gaussian:
        sigma = r/4
        params['gaussian_sigma'] = sigma
        data = np.exp( (data/sigma)**2/2 )
    dist_matrix = csr_matrix((data, (row, col)), shape=(nc, nc))

    adata.obsp[f'{spatial_key}_distances'] = dist_matrix.copy()
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

        sig_mat = csr_matrix((l_data*r_data/d,indices.copy(),indptr.copy()), shape=(nc, nc))# .copy() is necessary. eliminate_zeros() removes indices and indptr inplace
        sig_mat.eliminate_zeros()
        I[i] = sig_mat.nnz>0
        if I[i]:
            adata.obsp[lr_key] = sig_mat
            lr_keys.append(lr_key)

    # df_ligrec_detected = df_ligrec[I]
    adata.uns['LR_signal_info'] = {'signal': lr_keys, 'db': df_ligrec, 'params': params}

def LR_signal2(
        adata, 
        df_ligrec, 
        spatial_key = 'spatial', 
        heteromeric_delimiter = '_', 
        lr_delimiter = '-',
        key_added = 'LR',
        # gaussian = True
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

def plot_xc2grid(xc, xc2grid_map, grid_center, ax = None, show_axis = True):
    seg = []
    for ij in xc2grid_map:
        cid = xc2grid_map[ij]
        xc_ij =  xc[cid]
        xc_grid_ij = grid_center[ij]
        starts = np.repeat(xc_grid_ij.reshape(1,-1), len(cid), axis=0)
        seg.append(np.stack([starts, xc_ij], axis=1))
    seg = np.concatenate(seg)
    lc = LineCollection(seg, colors='k', linewidths=0.5)

    Xgrid = np.array(list(grid_center.values()))

    fig, ax = _get_axes(ax)
    ax.plot(xc[:,0],xc[:,1],'.',ms=5)
    ax.plot(Xgrid[:,0],Xgrid[:,1],'r.')
    ax.add_collection(lc)
    _set_axes(ax, show_axis)
    return ax

def plot_grid(x_grid, y_grid, ax = None):
    fig, ax = _get_axes(ax)
    for i in range(len(x_grid)):
        ax.plot(x_grid[[i,i]],y_grid[[0,-1]],'k:',lw=0.5)
    for i in range(len(y_grid)):
        ax.plot(x_grid[[0,-1]],y_grid[[i,i]],'k:',lw=0.5)
    return ax

def plot_signal_direction(
        Xgrid, Vgrid,
        rs = 'receiver',
        color = 'k',
        scale =1.,
        # pivot = 'tip',
        scale_units = 'x',
        ax=None,
):
    fig, ax = _get_axes(ax)
    if rs == 'receiver':
        ax.quiver(Xgrid[:,0], Xgrid[:,1], Vgrid[:,0], Vgrid[:,1], 
                color=color, angles='xy',scale=scale, pivot = 'tip', scale_units =scale_units)
    else:
        ax.quiver(Xgrid[:,0], Xgrid[:,1], Vgrid[:,0], Vgrid[:,1], 
                color=color, angles='xy',scale=scale, pivot = 'tail', scale_units =scale_units)
    return ax

def plot_signal_tensor(
        Xgrid, Vegrid,
        scale = 150,
        pivot = 'tail',
        headwidth = 3,
        headlength = 3.5,
        headaxislength = 3,
        width = 0.004,
        color = 'k',
        linestyle = '-',
        scale_units = 'width',
        th: float = 0.,
        ax=None
):
    fig, ax = _get_axes(ax)
    I = np.linalg.norm(Vegrid,axis=(1,2))>th
    for k in range(2):          # eigenvector index
        for sign in (1, -1):    # + and - direction
            ax.quiver(Xgrid[I, 0], Xgrid[I, 1], sign * Vegrid[I, 0, k], sign * Vegrid[I, 1, k],
                    color=color, angles='xy', pivot = pivot, scale = scale, scale_units = scale_units, linestyle = linestyle,
                    headwidth = headwidth, headlength = headlength, headaxislength = headaxislength, width=width)
    return ax

def _get_axes(
        ax = None,
        dim = 2
):
    """create or get axes"""
    if ax is None:
        if dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        else:
            fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax

def _set_axes(ax, show_axis):
    """aspect equal, axis off, invert yaxis"""
    ax.set_aspect('equal', adjustable='box')
    ax.autoscale(tight=True)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    if not show_axis:
        ax.set_axis_off()