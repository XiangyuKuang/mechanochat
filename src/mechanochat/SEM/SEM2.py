from typing import Optional, Union, List, Dict, Tuple
from math import pi, sqrt
from scipy.spatial import distance, KDTree
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix, csr_array
from matplotlib import colormaps
import numpy as np
from numpy.typing import NDArray
from numba import cuda, njit, prange
from pandas import DataFrame
import pickle
import os
from .SEM_utils import CellBase, find_pairs, d_potential_LJ_gpu, d_potential_LJ_gpu_r2, potential_LJ_gpu,  virial_stess_tensor
from .._utils import AlphaShape, scale
from anndata import AnnData
import warnings
from tqdm import tqdm

class SEM2(CellBase):
    """Subcellular Element Method (Dev)"""

    sim_name: str
    """simulation name"""
    t: int
    """time step""" 
    param: dict
    """simulation parameters (Deprecated)"""
    f_cc: csr_matrix
    """cell-cell force matrix (symmetric, abs)"""
    pairs: NDArray
    """
    interaction element pairs [i,j]
    
    pairs[k] = [i,j]
    """
    f_pairs: NDArray
    """
    f_pairs[k] stores force j -> i
    
    pairs[k] = [i,j]
    """

    def __init__(self, 
            ne_per_cell: int,
            re: float,
            rd_ratio: float,
            adata: AnnData,
            cluster_key: str = 'leiden',
            spatial_key: str = 'spatial',
            xc: Optional[NDArray] = None,
            ctype: Optional[NDArray] = None,
            sim_name: str = 'untitled',
            seed: int = 0
    ):
        """
        Create a SEM object

        Parameters
        -------
        ne_per_cell : int, default: 20
            Number of elements per cell
        re : float
            Element radius
        rd_ratio : float
            Cell radius-distance ratio

            rd_ratio>2: cell radius < cell distance/2, tissue with gaps

            rd_ratio=2: cell radius = cell distance/2, no gaps (confluent tissue)

            rd_ratio<2: cell radius > cell distance/2, overcrowded
        adata : Anndata
            Anndata with .obsm[spatial_key] for cell coordinates, .obs[cluster_key] for cell types, .obsm[embedding_key] for low-dim embedding

            If not provided, xc and ctype are required
        xc : Optional[np.ndarray]
            cell coordinates. Ignored, if adata.obsm[spatial_key] is provided
        ctype : Optional[np.ndarray]
            cell types.  Ignored, if adata.obs[cluster_key] is provided
        cluster_key : str, default: 'leiden'
            Key for cell type in .obs
        spatial_key : str, default: 'spatial'
            Key for spatial coordinates in .obsm
        embedding_key : str, default: 'X_pca'
            Key for low-dim embedding in .obsm, used for computing gene similarity
        sim_name : str, default: 'untitled'
            Simulation name
        """

        super().__init__(adata, xc, ctype, cluster_key, spatial_key, None)
        # check 3d
        if self.dim > 2:
            warnings.warn('xc is 3d')
            self.dim = 2
            self.xc = self.xc[:,[0,1]]

        # Initialize simulation-specific properties
        # cell radius
        if self.nc > 2:
            # estimate cell radius by Delaunay
            distance_matrix = self.compute_distance('delaunay', return_distances=True)
            indices = distance_matrix.indices
            indptr = distance_matrix.indptr
            dc = np.zeros(self.nc)
            for cid in range(self.nc):
                _,j=distance_matrix[cid].nonzero()
                dc[cid] = np.mean(distance_matrix[cid,j]) if len(j)>0 else np.nan # some points might overlap with others
            rc = np.median(dc)/rd_ratio
            # rd_ratio>2: cell radius < cell distance/2, tissue with gaps
            # rd_ratio=2: cell radius = cell distance/2, no gaps (confluent tissue)
            # rd_ratio<2: cell radius > cell distance/2, overcrowded

            ## cell sizes
            cell_size = dc/np.median(dc)
            # adjust boundary cells 
            hull = AlphaShape(self.xc,no_hole=False)
            Ib = np.unique(np.concatenate(hull.get_boundary_vertices1()))
            if self.nc > len(Ib): 
                cellsize_neig = np.zeros(len(Ib))
                for _ in range(10):
                    for n,cid in enumerate(Ib):
                        j = indices[indptr[cid]:indptr[cid+1]]
                        cellsize_neig[n] = np.median(cell_size[j])
                    Ib_oversize = np.where(cell_size[Ib]>cellsize_neig*1.2)[0]
                    I_oversize = Ib[Ib_oversize]
                    if len(I_oversize)==0:
                        break
                    else:
                        cell_size[I_oversize] = cellsize_neig[Ib_oversize]
            cell_size[cell_size>1.25] = 1.25
            
        elif self.nc == 2:
            # only two cells
            rc = distance.euclidean(self.xc[0],self.xc[1])/2
            cell_size = np.ones(2)
        else:
            # only one cell
            rc = 1
            cell_size = np.array([1])
        
        self.rc = rc
        self.cell_size = cell_size
        # rc_n = np.sqrt(ne_per_cell*(re/2)**2*np.sqrt(3)/2/np.pi)# cell radius in simulation, treat elements as hexgon
        rc_n = np.sqrt(ne_per_cell*(re/2)**2*3*np.sqrt(3)/2/np.pi)# cell radius in simulation, treat elements as hexgon
        # rc_n = np.sqrt(ne_per_cell*(re/2)**2)# cell radius in simulation, treat elements as circle
        self.scale = rc/rc_n
        self.deltax = np.mean(self.xc, axis=0)
        self.xc = (self.xc-self.deltax)/self.scale ## scaling xc to xc_n/rc_n = xc/rc
        
        ## element info
        self.ne_per_cell = (self.cell_size**2*ne_per_cell).astype(np.int32) # cellsize = 1 -> nepc = ne_per_cell
        self.ne = self.ne_per_cell.sum() # total number of elements
        self.ecid = np.repeat(np.arange(self.nc, dtype=np.int32), self.ne_per_cell) # id of cell to which each element belongs
        self.ceidn = np.insert(self.ne_per_cell.cumsum(),0,0)# first elements id of each cell. Elements of cell i can be retrieve by ceidn[i]:ceidn[i+1]. Number of elements of cell i is ceidn[i+1]-ceidn[i]
        self.xe = np.zeros((self.ne, self.dim), dtype=np.float32) # elements coordinates, n_element*dim
        ## random number generator
        self.rng = np.random.default_rng(seed)
        self.rng_seed = seed
        ## deploy elements to the spherical region around each cell coordinates
        for cid in range(self.nc):
            # generate element in a spherical region following uniform distribution
            ne_i = self.ceidn[cid+1] - self.ceidn[cid]
            xc_i = self.xc[cid]
            r = rc_n*self.cell_size[cid]*np.sqrt(self.rng.uniform(0, 1, size=(ne_i, 1))) #np.sqrt() # cell_r = rc_n*self.ne_per_cell[i]/ne_per_cell
            phi = self.rng.uniform(-pi, pi, size=(ne_i, 1))
            xe = np.concatenate((r*np.cos(phi), r*np.sin(phi)), axis=1)
            xe = xe-np.mean(xe, axis=0)+xc_i # move initial element to cell center
            # shrink elements to cell center, avoid cell-cell overlap
            if self.nc > 2:
                j = indices[indptr[cid]:indptr[cid+1]]
                xc_ij = np.insert(self.xc[j],0,xc_i,axis=0)
                I = distance.cdist(xc_ij,xe).argmin(axis=0) != 0
                while np.any(I):
                    xe[I] = (xe[I]-xc_i)*0.5+xc_i
                    I = distance.cdist(xc_ij,xe).argmin(axis=0) != 0
            self.xe[self.ceidn[cid]:self.ceidn[cid+1]] = xe.astype(np.float32)

        # simulation info
        self.sim_name = sim_name
        self.t = 0
        self.param = dict()
            
        self.ns_default = 10 # alphashape default param

    def get_param(
            self,
            df_adh: DataFrame,
            df_stiffness: DataFrame,
            embedding_key: str = 'X_pca',
            adh_range: Tuple[float, float] = (0.25,4.),
            stiffness_range: Tuple[float, float] = (0.25,0.5),
            w: float = 0.75,
            return_data: bool = False
    ):
        """Compute cell mechanical parameters alpha (adhesion) and beta (stiffness)
        from gene expression data and store them in ``self.alpha_csr`` and ``self.beta``.

        **Computation pipeline**

        1. **Neighbor detection** — build a KD-tree on cell centroids ``self.xc``
           and find all cell pairs within a Chebyshev distance of 60 (simulation units).
           Only neighboring pairs contribute to adhesion and similarity terms.

        2. **Adhesion score** (``adhesion_data``) — for each ligand-receptor (LR) pair
           in ``df_adh``, the per-cell ligand activity *l* and receptor activity *r* are
           computed as the product of min-shifted subunit expression levels (supports
           multi-subunit ligands/receptors encoded as ``'gene1_gene2'`` in the table).
           The symmetric interaction score for neighbor pair *(i, j)* is::

               LR_ij = r_i * l_j + l_i * r_j

           Scores from all LR pairs are summed, capped at the 95th percentile to reduce
           outlier influence, then scaled to ``adh_range``.

        3. **Gene expression similarity** (``sim_data``) — cosine similarity between
           cell embedding vectors (default: PCA, ``adata.obsm[embedding_key]``) for each
           neighbor pair, computed as::

               cos_sim_ij = (X_i · X_j) / (||X_i|| * ||X_j||)

           Values are scaled to ``adh_range``.

        4. **Alpha matrix** — the cell-cell adhesion parameter for pair *(i, j)* is a
           weighted combination of expression similarity and LR-based adhesion::

               alpha_ij = sim_data_s * (1 - a) + adhesion_data * a

           Stored as a symmetric sparse matrix ``self.alpha_csr`` of shape
           ``(nc, nc)``.

        5. **Beta vector** — mean expression of cytoskeleton-related genes in
           ``df_stiffness`` per cell, capped at the 95th percentile and scaled to
           ``stiffness_range``. Stored as ``self.beta`` of shape ``(nc,)``.

        Parameters
        ----------
        df_adh : DataFrame
            Ligand-receptor pair table. Column 0 = ligand gene(s), column 1 = receptor
            gene(s). Multi-subunit complexes should be joined with ``'_'``
            (e.g. ``'ITGA1_ITGB1'``).
        df_stiffness : DataFrame
            Cytoskeleton/stiffness gene table. Column 0 contains gene names whose mean
            expression determines per-cell stiffness.
        embedding_key : str, optional
            Key in ``adata.obsm`` for the cell embedding used to compute gene expression
            similarity. Default ``'X_pca'``.
        adh_range : tuple of float, optional
            ``(min, max)`` range that both ``adhesion_data`` and ``sim_data`` are scaled
            to before combining. Default ``(0.25, 4.0)``.
        stiffness_range : tuple of float, optional
            ``(min, max)`` range that stiffness values are scaled to. Default
            ``(0.25, 0.5)``.
        w : float, optional
            Mixing weight for adhesion vs. similarity in the alpha parameter.
            ``w=1`` → pure LR adhesion; ``w=0`` → pure gene expression similarity.
            Default ``0.75``.
        return_data : bool, optional
            If ``True``, return intermediate arrays for inspection. Default ``False``.

        Returns
        -------
        None
            Sets ``self.alpha_csr`` (sparse ``(nc, nc)`` float32) and ``self.beta``
            (``(nc,)`` float32) in place.
        row_indices, col_indices, lr, sim_data, stiffness_data : only when ``return_data=True``

            ``row_indices``, ``col_indices``: COO indices of neighbor pairs (symmetric).

            ``lr``: raw (unscaled) symmetric LR adhesion scores.

            ``sim_data``: raw symmetric cosine similarity values.

            ``stiffness_data``: raw stiffness scores.
        """
        # --- Step 1: find neighboring cell pairs within Chebyshev distance r=60 ---
        tree = KDTree(self.xc)
        r = 60
        pairs = tree.query_pairs(r, p=np.inf, output_type='ndarray')
        n_pairs = pairs.shape[0]
        i_indices = pairs[:, 0]
        j_indices = pairs[:, 1]

        # --- Step 2: compute LR-based adhesion scores for all neighbor pairs ---
        lr_values = np.zeros(n_pairs, dtype=np.float32)
        for idx in range(df_adh.shape[0]):
            l_gene = df_adh.iloc[idx, 0]        # ligand (may be 'geneA_geneB' for complex)
            r_gene = df_adh.iloc[idx, 1]   # receptor (may be 'geneC_geneD' for complex)

            # min-shift each subunit before taking the product across subunits,
            # so that a zero-expressed subunit zeroes out the complex activity
            l_temp = self.adata[:, l_gene.split('_')].X.toarray()
            r_temp = self.adata[:, r_gene.split('_')].X.toarray()
            l_data = np.prod(l_temp - l_temp.min(axis=0), axis=1)
            r_data = np.prod(r_temp - r_temp.min(axis=0), axis=1)

            # symmetric interaction: cell i can be sender or receiver
            lr_values += (r_data[i_indices] * l_data[j_indices] + l_data[i_indices] * r_data[j_indices])

        # mirror upper-triangle pairs to build a symmetric COO representation
        row_indices = np.concatenate([i_indices, j_indices])
        col_indices = np.concatenate([j_indices, i_indices])
        lr = np.concatenate([lr_values, lr_values])

        # cap at 95th percentile to reduce outlier influence, then scale to adh_range
        high_lr = np.percentile(lr, 95)
        if high_lr == 0:
            adhesion_data = lr
        else:
            adhesion_data = np.minimum(lr, high_lr)
        adhesion_data_s, adh_a, adh_b = scale(adhesion_data, adh_range)

        # --- Step 3: compute gene expression similarity (cosine similarity in embedding space) ---
        X_em = self.adata.obsm[embedding_key]  # shape (nc, n_components)
        norms = np.linalg.norm(X_em, axis=1)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        dot_products = np.sum(X_em[i_indices] * X_em[j_indices], axis=1)
        cos_sim_values = (dot_products / (norms[i_indices] * norms[j_indices])).astype(np.float32)
        sim_data = np.concatenate([cos_sim_values, cos_sim_values])
        sim_data_s, sim_a, sim_b = scale(sim_data, adh_range)

        # --- Step 4: combine similarity and adhesion into alpha_csr ---
        # alpha_ij = sim_data_s * (1 - a) + adhesion_data * a
        self.alpha_csr = csr_matrix(
            (sim_data_s*(1-w)+adhesion_data_s*w, (row_indices, col_indices)),
            shape=(self.nc, self.nc),
            dtype=np.float32
        )

        # --- Step 5: compute per-cell stiffness beta from cytoskeleton gene expression ---
        stiffness_exp = self.adata[:, df_stiffness.iloc[:, 0]].X.toarray()  # (nc, n_genes)
        stiffness_exp_m = stiffness_exp.mean(axis=1)                         # mean across genes
        high_stiffness = np.percentile(stiffness_exp_m, 95)
        if high_stiffness == 0:
            stiffness = stiffness_exp_m
        else:
            stiffness = np.minimum(stiffness_exp_m, high_stiffness)
        stiffness, stiff_a, stiff_b = scale(stiffness, stiffness_range)
        self.beta = stiffness.astype(np.float32)
        self.param_info = {
            'adh_a':adh_a,
            'adh_b':adh_b,
            'high_lr': high_lr,
            'sim_a': sim_a, 
            'sim_b': sim_b,
            'w': w,
            'stiff_a': stiff_a,
            'stiff_b': stiff_b,
            'high_stiffness': high_stiffness,
        }
        print('add .beta .alpha_csr .param_info')
        if return_data:
            return row_indices, col_indices, lr, sim_data, stiffness_exp_m

    def sim_gpu(self,rm_intra,rm_inter,sigma,dt,T,enable_tqdm=True) -> None:
        # update: gpu thread number, pre-compute rm^6, call dynamics2d_gpu4_SEM2_opt()

        # test pre defined parameters alpha,beta,sigma,gamma
        alpha = self.alpha_csr.data.copy()
        alpha_indices = self.alpha_csr.indices.copy()
        alpha_indptr = self.alpha_csr.indptr.copy()

        self.param["rm_intra"] = rm_intra
        self.param["rm_inter"] = rm_inter
        rm6_intra = rm_intra**6
        rm6_inter = rm_inter**6
        # d_rm_intra = cupy.asanyarray(rm_intra*self.cell_size)
        d_xe = cuda.to_device(self.xe)
        d_xe_F = cuda.to_device(self.xe)
        d_ecid = cuda.to_device(self.ecid)
        d_alpha = cuda.to_device(alpha)
        d_alpha_indices = cuda.to_device(alpha_indices)
        d_alpha_indptr = cuda.to_device(alpha_indptr)
        d_ceidn = cuda.to_device(self.ceidn)
        d_beta = cuda.to_device(self.beta)
        sigmadt = sqrt(dt) * sigma
        # gpu thread number
        tpb = 128
        bpg = max((self.ne+tpb-1)//tpb,128)
        # print(tpb,bpg)
        # iteration
        cuda.synchronize()
        for t in tqdm(range(T),'Simulation',disable=not enable_tqdm):
            # a = np.sqrt((T-t)/T)
            # a = (1-t/T)**2
            a = 1
            # d_gamma = cuda.to_device(a*gamma)
            x_randt = cuda.to_device((sigmadt * a * self.rng.normal(0, 1, size=self.xe.shape)).astype(np.float32))#*self.cell_size[self.ecid,np.newaxis]#
            dynamics2d_gpu4_SEM2_opt[bpg, tpb](d_xe, d_xe_F, d_ecid, d_alpha, d_alpha_indices, d_alpha_indptr, d_ceidn, d_beta, x_randt, rm6_intra, rm6_inter, dt)# type: ignore
            cuda.synchronize()
            # var:t-1, var_F:t
            d_xe[:, :] = d_xe_F # update xe to t
            cuda.synchronize()
            # if self.t % vis_interval ==0:
            #     print(self.t)
            self.t += 1
        # close
        cuda.synchronize()
        self.xe = d_xe.copy_to_host()
        self.update_xc()
        self.alphashape_info['computed'] = False # marks alpha shapes need to be updated
    
    def compute_force(self, chunk_size:int = 300000000) -> None:
        """
        Compute inter/intracellular force
        
        add self.pairs[k] = [i,j]

        add self.f_pairs[k] stores force j -> i

        call `ee_pairwise_forces1_gpu_SEM1`

        Parameter
        ------
        """
        rm_intra = self.param["rm_intra"]
        rm_inter = self.param['rm_inter']

        pairs = find_pairs(self.xe,30) # 5.5s
        self.pairs = pairs
        n_pairs = pairs.shape[0]
        f_pairs = np.zeros((n_pairs,self.dim),dtype=np.float32)
        print('add .pairs')

        ecid0 = self.ecid[self.pairs[:,0]]
        ecid1 = self.ecid[self.pairs[:,1]]
        pair_alpha = self.alpha_csr[ecid0,ecid1].A1

        d_xe = cuda.to_device(self.xe) # float32
        d_ecid = cuda.to_device(self.ecid)
        d_beta = cuda.to_device(self.beta)

        # gpu thread number
        tpb = 128
        bpg = max((self.ne+tpb-1)//tpb,128)
        num_chunks = (n_pairs + chunk_size - 1) // chunk_size
        # computation
        for i in tqdm(range(num_chunks),'Computing force by chunk'):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_pairs)
            current_chunk_size = end_idx - start_idx
            d_pairs_chunk = cuda.to_device(pairs[start_idx:end_idx])
            f_pairs_chunk = cuda.to_device(np.zeros((current_chunk_size, self.dim), dtype=np.float32))
            d_pair_alpha_chunk = cuda.to_device(pair_alpha[start_idx:end_idx])
            cuda.synchronize()
            ee_pairwise_forces1_gpu_SEM2[bpg, tpb](# type: ignore
                d_xe, d_pairs_chunk, f_pairs_chunk, d_ecid, d_pair_alpha_chunk, d_beta, rm_intra, rm_inter
            )
            cuda.synchronize()
            f_pairs[start_idx:end_idx] = f_pairs_chunk.copy_to_host()
            del d_pairs_chunk, f_pairs_chunk
        self.f_pairs = f_pairs
        print('add .f_pairs')
        
        self.compute_cell_forces() # cell-cell force abs symmetric 

    def compute_element_energy(self):
        rm_intra = self.param["rm_intra"]
        rm_inter = self.param['rm_inter']
        d_xe = cuda.to_device(self.xe) # float32
        d_ecid = cuda.to_device(self.ecid)
        d_ceidn = cuda.to_device(self.ceidn)
        d_beta = cuda.to_device(self.beta)

        alpha = self.alpha_csr.data.copy()
        alpha_indices = self.alpha_csr.indices.copy()
        alpha_indptr = self.alpha_csr.indptr.copy()
        d_alpha = cuda.to_device(alpha)
        d_alpha_indices = cuda.to_device(alpha_indices)
        d_alpha_indptr = cuda.to_device(alpha_indptr)

        d_he = cuda.to_device(np.zeros(self.ne, dtype=np.float32))
        # gpu thread number
        tpb = 128
        bpg = max((self.ne+tpb-1)//tpb,128)
        # computation
        cuda.synchronize()
        e_energy_gpu[bpg, tpb](# type: ignore
            d_xe, d_he, d_ecid, d_ceidn, d_alpha, d_alpha_indices, d_alpha_indptr, d_beta, rm_intra, rm_inter,
        )
        cuda.synchronize()
        self.he = d_he.copy_to_host()
        print('add .he')

    def compute_cell_forces(self) -> None:
        """
        add self.f_cc: cell-cell force matrix (symmetric, abs)
        """
        indptr = self.alpha_csr.indptr.copy()
        indices = self.alpha_csr.indices.copy()
        nnz = self.alpha_csr.nnz
        nc = self.nc

        pair_csr_idx = pair2csr_map(self.pairs, self.ecid, indptr, indices)
        f_abs = np.linalg.norm(self.f_pairs,axis=1)
        f_cc_data = cell_cell_forces_csr(f_abs,pair_csr_idx,nnz)

        f_cc = csr_matrix((f_cc_data,indices,indptr),shape=(nc,nc))
        I = f_cc.data<1e-1
        f_cc.data[I] = 0
        f_cc.eliminate_zeros()
        print('add .f_cc')
        self.f_cc = f_cc
        if self.adata is not None:
            print("add .obsp['f_cc'] to adata")
            self.adata.obsp['f_cc'] = f_cc

    def f_pairs_tocsr(self, e_mask: Optional[NDArray[np.bool_]] = None) -> Tuple[csr_array, csr_array]:
        """
        Convert force pairs to CSR sparse matrices. Run `.compute_force()` first
        
        fx_csr[i, j] = fx_{j->i}
        
        row index i : under force
        column index j : exert force
        
        Parameters
        ----------
        e_mask : Optional[NDArray[np.bool_]]
            Boolean mask of shape (ne,) to select subset of elements.
            If provided, only interactions involving selected elements are included,
            and the returned matrices have shape (n_selected, n_selected) with
            re-indexed rows/columns.
            If None, return full matrices of shape (ne, ne).
        
        Returns
        -------
        fx_csr : csr_array
            Sparse matrix of x-component forces
        fy_csr : csr_array
            Sparse matrix of y-component forces
        """
        if not hasattr(self,'pairs'):
            RuntimeError('no .pairs, run .compute_force() first')
        
        if e_mask is None:
            # 全部elements
            i_idx = np.concatenate([self.pairs[:, 0], self.pairs[:, 1]])
            j_idx = np.concatenate([self.pairs[:, 1], self.pairs[:, 0]])
            fx_vals = np.concatenate([self.f_pairs[:, 0], -self.f_pairs[:, 0]])
            fy_vals = np.concatenate([self.f_pairs[:, 1], -self.f_pairs[:, 1]])
            ne = self.ne
        else:
            # Subset模式：只处理e_mask中的elements
            # 创建索引映射: 原始index -> 新index
            e_indices = np.where(e_mask)[0]
            ne = len(e_indices)
            old2new = -np.ones(self.ne, dtype=np.int64)
            old2new[e_indices] = np.arange(ne)
            
            # 筛选pairs: 只保留两端都在e_mask中的pairs
            pair_mask = e_mask[self.pairs[:, 0]] & e_mask[self.pairs[:, 1]]
            pairs_sub = self.pairs[pair_mask]
            f_pairs_sub = self.f_pairs[pair_mask]
            
            # 映射到新索引
            i_idx = np.concatenate([old2new[pairs_sub[:, 0]], old2new[pairs_sub[:, 1]]])
            j_idx = np.concatenate([old2new[pairs_sub[:, 1]], old2new[pairs_sub[:, 0]]])
            fx_vals = np.concatenate([f_pairs_sub[:, 0], -f_pairs_sub[:, 0]])
            fy_vals = np.concatenate([f_pairs_sub[:, 1], -f_pairs_sub[:, 1]])
        
        fx_coo = coo_matrix((fx_vals, (i_idx, j_idx)), shape=(ne, ne))
        fy_coo = coo_matrix((fy_vals, (i_idx, j_idx)), shape=(ne, ne))
        fx_csr = fx_coo.tocsr()
        fy_csr = fy_coo.tocsr()
        
        if not (np.array_equal(fx_csr.indices, fy_csr.indices) and 
                np.array_equal(fx_csr.indptr, fy_csr.indptr)):
            warnings.warn('fx_csr, fy_csr diff')
        
        return fx_csr, fy_csr

    def compute_tensor(self) -> None:
        """
        compute element stress tensor

        run compute_force first

        add .e_sigma
        """
        fx_csr, fy_csr = self.f_pairs_tocsr()
        indptr  = fx_csr.indptr    # (ne+1,)
        indices = fx_csr.indices   # (nnz_total,)
        fx_data = fx_csr.data      # (nnz_total,)
        fy_data = fy_csr.data      # (nnz_total,)
        self.e_sigma = virial_stess_tensor(indptr, indices, fx_data, fy_data, self.xe)
        print('add .e_sigma')

    def compute_pressure(self) -> None:
        """
        compute element pressure

        run compute_tensor first

        add .p
        """
        if not hasattr(self,'e_sigma'):
            self.compute_tensor()
        p = np.zeros(self.ne)
        for i in range(self.ne):
            p[i] = -np.trace(self.e_sigma[i])/2
        self.p = p
        print('add .p')

    def _get_e_radius(self) -> float:
        """get SEM default d_th for computing contact and alphashape"""
        if len(self.param) > 0:
            d_th = self.param['rm_inter']
        else:
            warnings.warn('rm_inter is not provided, using d_th = 1')
            d_th = 1.0
        return d_th

    def save_sim(self) -> None:
        """
        Save simulation
        """
        filename = f'{self.sim_name}_{self.t}'
        if os.path.exists(filename+'.pkl'):
            warnings.warn(f"File '{filename}' already exists.")
            filename = filename + '_temp'
        filename = filename+'.pkl'

        with open(filename, 'wb') as f:
            data = {
                'xe': self.xe,
                'ceidn': self.ceidn,
                'ecid': self.ecid,
                'param': self.param,
                'scale': self.scale,
                'deltax': self.deltax
            }
            if hasattr(self,'pairs'):
                data['pairs'] = self.pairs
                data['f_pairs'] = self.f_pairs
            if hasattr(self,'beta'):
                data['beta'] = self.beta
            if hasattr(self,'alpha_csr'):
                data['alpha_csr'] = self.alpha_csr
            pickle.dump(data, f)
        print(f"saved as {filename}")

    def load_sim(self, sim_name: str , t: int, path: str = '.', rename: bool = True) -> None:
        '''
        Restore a simulation from `{path}/{sim_name}_{t}.pkl`

        Parameters
        ------
        sim_name : str
            Name of simulation
        t : int
            Time point
        path : str, default: '.'
            Path to simulation data
        rename : bool, default: True
            If True, rename the `sem` to `sim_name`
        '''
        filename = f'{path}/{sim_name}_{t}.pkl'
        with open(filename, 'rb') as f:
            print(f'load sim data from {filename}')
            data = pickle.load(f)
            self.xe = data['xe']
            self.ceidn = data['ceidn']
            self.ecid = data['ecid']
            self.scale = data['scale']
            self.deltax = data['deltax']
            if 'param' in data:
                print('.param loaded')
                self.param = data['param']
            if 'beta' in data:
                print('.beta loaded')
                self.beta = data['beta']
            if 'alpha_csr' in data:
                print('.alpha_csr loaded')
                self.alpha_csr = data['alpha_csr']
            if 'pairs' in data:
                print('.pairs loaded\n.f_pairs loaded')
                self.pairs = data['pairs']
                self.f_pairs = data['f_pairs']
                if hasattr(self,'alpha_csr'):
                    self.compute_cell_forces()
            
        self.t = t
        self.update_xc()
        if rename:
            print(f'Simulation renamed as {sim_name}')
            self.sim_name = sim_name

@cuda.jit
def dynamics2d_gpu4_SEM2_opt(xe, xe_F, ecid, alpha, alpha_indices, alpha_indptr, ceidn, beta, x_randt, rm6_intra, rm6_inter, dt):
    """
    dynamics2d_gpu2

    add beta_i for each cell

    inter dV = max(beta[cid]*d_potential_LJ_gpu(r, rm_intra, 1.5) + gamma[cid]/r**3, -10.0 )

    intra dV = max(alpha[cid,ecid[j]]*d_potential_LJ_gpu(r, rm_inter, 1.5), -10.0)
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    ne = xe.shape[0]
    for i in range(start, ne, stride):
        xi = xe[i, 0]
        yi = xe[i, 1]
        cid = ecid[i]
        beta_i = beta[cid]
        fx = 0.0
        fy = 0.0

        # Intra-cell interactions
        intra_start = ceidn[cid]
        intra_end = ceidn[cid + 1]
        for j in range(intra_start, intra_end):
            if j == i :
                continue
            xj = xe[j, 0]
            yj = xe[j, 1]
            deltax = xi - xj
            deltay = yi - yj
            deltax_s = deltax*deltax
            deltay_s = deltay*deltay
            r2 = deltax_s + deltay_s
            dV = beta_i*d_potential_LJ_gpu_r2(r2, rm6_intra, 1.)
            if dV < -10.0:
                dV = -10.0
            fx -= dV * deltax
            fy -= dV * deltay

        # Inter-cell interactions
        neighbor_start = alpha_indptr[cid]
        neighbor_end = alpha_indptr[cid + 1]
        for k in range(neighbor_start, neighbor_end):
            cid_j = alpha_indices[k]
            alpha_ij = alpha[k]
            inter_start = ceidn[cid_j]
            inter_end = ceidn[cid_j + 1]
            for j in range(inter_start, inter_end):
                xj = xe[j, 0]
                yj = xe[j, 1]
                deltax = xi - xj
                deltay = yi - yj
                deltax_s = deltax*deltax
                deltay_s = deltay*deltay
                if deltax_s < 900 and deltay_s < 900:
                    r2 = deltax_s + deltay_s
                    dV = alpha_ij*d_potential_LJ_gpu_r2(r2, rm6_inter, 1.)
                    if dV < -10.0:
                        dV = -10.0
                    fx -= dV * deltax
                    fy -= dV * deltay
        ## 每次迭代都乘以 dt
        #         xe_F[i, 0] += -dt * dV * deltax
        #         xe_F[i, 1] += -dt * dV * deltay
        # xe_F[i, 0] += x_randt[i, 0]
        # xe_F[i, 1] += x_randt[i, 1]
        ## 最后统一乘以 dt，浮点运算不等价

        xe_F[i, 0] += dt * fx + x_randt[i, 0]
        xe_F[i, 1] += dt * fy + x_randt[i, 1]
        ##

@cuda.jit
def ee_pairwise_forces1_gpu_SEM2(xe, pairs, pair_force, ecid, pair_alpha, beta, rm_intra, rm_inter):
    """
    dynamics2d_gpu4_SEM1

    inter dV = beta[cid_i] * d_potential_LJ_gpu(r, rm_intra, 1.5)

    intra dV = alpha[cid_i, cid_j] * d_potential_LJ_gpu(r, rm_inter, 1.5)

    对预计算得到的每个 element pair 计算作用力，结果存储在 pair_force 数组中，每行保存 (fx, fy)
    
    pair[1]施加给pair[0]的力: pair[1] -> pair[0]

    Parameters
    ------
      xe         : (ne, 2) 的 element 坐标数组
      pairs      : (num_pairs, 2) 的预计算 pair 索引数组，每行 (i, j)
      pair_force : (num_pairs, 2) 的输出数组，存储每个 pair j->i的力
      ecid       : (ne,) 数组，记录每个 element 属于哪个细胞
      alpha      : (nc, nc) 数组，不同细胞间的相互作用参数
      beta       : (nc,) 数组，同一细胞内相互作用参数
      gamma      : (nc,) 数组，同一细胞内额外的刚性参数
      rm_intra   : (nc,) 数组，每个细胞内部的 rm 参数 (通过细胞ID索引)
      rm_inter   : 标量，不同细胞间的 rm 参数
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n_pairs = pairs.shape[0]
    for k in range(start, n_pairs, stride):
        i = pairs[k, 0]
        j = pairs[k, 1]
        alpha_ij = pair_alpha[k]
        deltax = xe[i, 0]-xe[j, 0]
        deltay = xe[i, 1]-xe[j, 1]
        r = sqrt(deltax**2 + deltay**2)
        cid_i = ecid[i]
        cid_j = ecid[j]
        beta_i = beta[cid_i]
        if cid_j == cid_i:
            dV = beta_i * d_potential_LJ_gpu(r, rm_intra, 1.)
        else:
            dV = alpha_ij * d_potential_LJ_gpu(r, rm_inter, 1.)

        if dV < -10.0:
            dV = -10.0
            
        pair_force[k, 0] = -dV * deltax
        pair_force[k, 1] = -dV * deltay

@cuda.jit
def e_energy_gpu(xe, he, ecid, ceidn, alpha, alpha_indices, alpha_indptr, beta, rm_intra, rm_inter):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    ne = xe.shape[0]
    for i in range(start, ne, stride):
        xi = xe[i, 0]
        yi = xe[i, 1]
        cid = ecid[i]
        beta_i = beta[cid]
        he_i = 0

        # Intra-cell interactions
        intra_start = ceidn[cid]
        intra_end = ceidn[cid + 1]
        for j in range(intra_start, intra_end):
            if j == i :
                continue
            xj = xe[j, 0]
            yj = xe[j, 1]
            deltax = xi - xj
            deltay = yi - yj
            deltax_s = deltax*deltax
            deltay_s = deltay*deltay
            r = sqrt(deltax_s + deltay_s)
            he_i += beta_i*potential_LJ_gpu(r, rm_intra, 1.)
        
        # Inter-cell interactions
        neighbor_start = alpha_indptr[cid]
        neighbor_end = alpha_indptr[cid + 1]
        for k in range(neighbor_start, neighbor_end):
            cid_j = alpha_indices[k]
            alpha_ij = alpha[k]
            inter_start = ceidn[cid_j]
            inter_end = ceidn[cid_j + 1]
            for j in range(inter_start, inter_end):
                xj = xe[j, 0]
                yj = xe[j, 1]
                deltax = xi - xj
                deltay = yi - yj
                deltax_s = deltax*deltax
                deltay_s = deltay*deltay
                if deltax_s < 900 and deltay_s < 900:
                    r = sqrt(deltax_s + deltay_s)
                    he_i += alpha_ij*potential_LJ_gpu(r, rm_inter, 1.)
        
        he[i] = he_i

@njit()
def pair2csr_map(pairs, ecid, indptr, indices):
    """
    为每个inter-cell pair预计算其在CSR中的位置
    返回: pair_csr_idx (m, 2), 存储每个pair对应的两个CSR data索引
          -1表示intra-cell pair
    """
    m = pairs.shape[0]
    pair_csr_idx = np.full((m, 2), -1, dtype=np.int32)
    
    for k in range(m):
        e1 = pairs[k, 0]
        e2 = pairs[k, 1]
        c1 = ecid[e1]
        c2 = ecid[e2]
        
        if c1 != c2:
            # 找(c1, c2)在CSR中的位置
            for idx in range(indptr[c1], indptr[c1 + 1]):
                if indices[idx] == c2:
                    pair_csr_idx[k, 0] = idx
                    break
            
            # 找(c2, c1)在CSR中的位置
            for idx in range(indptr[c2], indptr[c2 + 1]):
                if indices[idx] == c1:
                    pair_csr_idx[k, 1] = idx
                    break
    
    return pair_csr_idx

@njit()
def cell_cell_forces_csr(f_abs, pair_csr_idx, nnz):
    """
    使用预计算的映射快速累加
    """
    data = np.zeros(nnz, dtype=f_abs.dtype)
    m = f_abs.shape[0]
    
    for k in range(m):
        idx0 = pair_csr_idx[k, 0]
        if idx0 >= 0:  # inter-cell pair
            v = f_abs[k]
            data[idx0] += v
            data[pair_csr_idx[k, 1]] += v
    
    return data