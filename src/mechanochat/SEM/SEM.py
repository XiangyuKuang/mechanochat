from typing import Optional, Tuple, Union, List, Dict
from math import pi, sqrt
from scipy.spatial import Delaunay, distance
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix, csr_array
import numpy as np
from numpy.typing import NDArray
from numba import cuda
import pickle
import os
from .SEM_utils import (
    CellBase,
    find_pairs,
    d_potential_LJ_gpu, 
    d_potential_LJ_gpu_r2,
    d_potential_LJ_cpu,
    cell_cell_forces,
    virial_stess_tensor
)
from .._utils import AlphaShape
from anndata import AnnData
import warnings
from tqdm import tqdm
# warnings.simplefilter('always', category=Warning)
# warnings.filterwarnings('error', category=RuntimeWarning)

class SEM(CellBase):
    """Subcellular Element Method (Deprecated)"""

    sim_name: str
    """simulation name"""
    t: int
    """time step""" 
    param: dict
    """simulation parameters"""

    def __init__(self, 
        ne_per_cell: int,
        re: float,
        rd_ratio: float = 2,
        adata: Optional[AnnData] = None,
        cluster_key: str = 'leiden',
        spatial_key: str = 'spatial',
        embedding_key: str = 'X_pca',
        xc: Optional[NDArray] = None,
        ctype: Optional[NDArray] = None,
        sim_name: str = 'untitled',# param: dict = {}
        seed: int = 1
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
        warnings.warn('SEM is deprecated, use `SEM1` instead',DeprecationWarning)
        super().__init__(adata, xc, ctype, cluster_key, spatial_key, None)
        # Initialize simulation-specific properties
        # element info
        self.ne = ne_per_cell * self.nc # total number of elements
        self.ecid = np.repeat(np.arange(self.nc, dtype=np.int32), ne_per_cell) # id of cell to which each element belongs
        self.ceidn = np.insert(np.cumsum([ne_per_cell]*self.nc), 0, 0)# first elements id of each cell. Elements of cell i can be retrieve by ceidn[i]:ceidn[i+1]. Number of elements of cell i is ceidn[i+1]-ceidn[i]
        self.xe = np.zeros((self.ne, self.dim), dtype=np.float32) # elements coordinates, n_element*dim
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
        elif self.nc == 2:
            # only two cells
            rc = distance.euclidean(self.xc[0],self.xc[1])/2
        else:
            # only one cell
            rc = 1
        self.rc = rc
        rc_n = np.sqrt(ne_per_cell*(re/2)**2*3*np.sqrt(3)/2/np.pi)# cell radius in simulation, treat elements as hexgon
        # rc_n = np.sqrt(ne_per_cell*(re/2)**2)# cell radius in simulation, treat elements as circle
        self.scale = rc/rc_n
        self.deltax = np.mean(self.xc, axis=0)
        self.xc = (self.xc-self.deltax)/self.scale ## scaling xc to xc_n/rc_n = xc/rc
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
        self.cell_size = cell_size
        # self.cell_size[self.cell_size>1.2] = 1.2
        # self.cell_size = np.ones(self.nc, dtype=np.float32)
        
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

        # adhesion based on gene simarity
        if self.adata is None:
            self.corr_matrix = np.ones((self.nc,self.nc), dtype=np.float32)
        else:
            X_em = self.adata.obsm[embedding_key]# default PCA matrix
            corr_matrix = np.corrcoef(X_em)
            c_min = 0.05
            corr_matrix[corr_matrix<c_min] = c_min
            self.corr_matrix = corr_matrix.astype(np.float32) # gene simarity matrix, (n_c*n_c)
            
        self.ns_default = 10 # alphashape default param
    
    def __repr__(self):
        return f'Simulation Name: {self.sim_name}\nt: {self.t}\nCell Number: {self.nc}\nElement Number: {self.xe.shape[0]}\nDim: {self.dim}\nParameters: {self.param}\nContact Matrix: {self.contact_matrix.__repr__()}'

    def _get_e_radius(self) -> float:
        """get SEM default d_th for computing contact and alphashape"""
        if len(self.param) > 0:
            d_th = self.param['rm_inter']
        else:
            warnings.warn('rm_inter is not provided, using d_th = 1')
            d_th = 1.0
        return d_th
    
    def sim_gpu(self, param: dict, T: int) -> None:
        """
        Implement SEM simulation

        Parameters
        ------
        param : dict
            Parameters
        T : int
            Time steps
        """
        self.param = param
        # get parameters
        rm_intra = param["rm_intra"]*self.cell_size#1.5
        rm_inter = param["rm_inter"]#3
        dt = param["dt"]#0.01
        sigma = param["sigma"]#0.25
        gamma = param["gamma"]#0.01
        alpha_max,alpha_min = param["alpha"]#0.25
        
        cmax = self.corr_matrix.max()
        cmin = self.corr_matrix.min()
        if cmax==cmin:
            # corr_matrix is constant, set alpha to ones
            alpha = alpha_max*np.ones_like(self.corr_matrix)
        else:
            # scale corr_matrix to [alpha_min,alpha_max]
            alpha = (alpha_max-alpha_min)/(cmax-cmin)*self.corr_matrix+(alpha_min*cmax-alpha_max*cmin)/(cmax-cmin)
        sigmadt = sqrt(dt) * sigma

        # transfer array to gpu
        d_rm_intra = cupy.asanyarray(rm_intra)
        d_xe = cupy.asanyarray(self.xe)
        d_xe_F = cupy.asanyarray(self.xe)
        d_ecid = cuda.to_device(self.ecid)
        d_alpha = cuda.to_device(alpha)

        # gpu thread number
        tpb = 128
        bpg = 128
        # iteration
        cuda.synchronize()
        for t in range(T):
            x_randt = cuda.to_device((sigmadt*np.sqrt((T-t)/T) * self.rng.normal(0, 1, size=self.xe.shape)).astype(np.float32))#*self.cell_size[self.ecid,np.newaxis]
            dynamics2d_gpu2[bpg, tpb](d_xe, d_xe_F, d_ecid, d_alpha, gamma, x_randt, d_rm_intra, rm_inter, dt)
            cuda.synchronize()
            # var:t-1, var_F:t
            d_xe[:, :] = d_xe_F # update xe to t
            cuda.synchronize()
            # if self.t % vis_interval ==0:
            #     print(self.t)
            self.t += 1
        # close
        cuda.synchronize()
        self.xe = d_xe.get()
        self.update_xc()
        self.alphashape_info['computed'] = False # marks alpha shapes need to be updated

    def sim_gpu_test(self,rm_intra,rm_inter,alpha,beta,gamma,sigma,dt,T) -> None:
        # test pre defined parameters alpha,beta,sigma,gamma
        self.param["rm_intra"] = rm_intra
        self.param["rm_inter"] = rm_inter
        d_rm_intra = cupy.asanyarray(rm_intra*self.cell_size)
        d_xe = cuda.to_device(self.xe)
        d_xe_F = cuda.to_device(self.xe)
        d_ecid = cuda.to_device(self.ecid)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        d_alpha = cuda.to_device(alpha.astype(np.float32))
        d_beta = cuda.to_device(beta.astype(np.float32))
        d_gamma = cuda.to_device(gamma.astype(np.float32))
        sigmadt = sqrt(dt) * sigma
        # gpu thread number
        tpb = 128
        bpg = max((self.ne+tpb-1)//tpb,128)
        print(tpb,bpg)
        # iteration
        cuda.synchronize()
        for t in range(T):
            x_randt = cuda.to_device((sigmadt*np.sqrt((T-t)/T) * self.rng.normal(0, 1, size=self.xe.shape)).astype(np.float32))#*self.cell_size[self.ecid,np.newaxis]#
            dynamics2d_gpu3[bpg, tpb](d_xe, d_xe_F, d_ecid, d_alpha, d_beta, d_gamma, x_randt, d_rm_intra, rm_inter, dt)
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

    def compute_force(self, method: str) -> None:
        """
        Compute inter/intracellular force
        
        Add self.xe_F_intra, self.xe_F_inter

        Parameter
        ------
        method : str
            'e' for force on element;  'ee' for element-element force
        """
        # self.param = param
        # # get parameters
        # rm_intra = param["rm_intra"]*self.cell_size#1.5
        # rm_inter = param["rm_inter"]#3
        # gamma = param["gamma"]#0.01
        # alpha_max,alpha_min = param["alpha"]#0.25
        
        # cmax = self.corr_matrix.max()
        # cmin = self.corr_matrix.min()
        # if cmax==cmin:
        #     # corr_matrix is constant, set alpha to ones
        #     alpha = np.ones_like(self.corr_matrix)
        # else:
        #     # scale corr_matrix to [alpha_min,alpha_max]
        #     alpha = (alpha_max-alpha_min)/(cmax-cmin)*self.corr_matrix+(alpha_min*cmax-alpha_max*cmin)/(cmax-cmin)
        # if lr is not None:
        #     alpha+=lr

        # get parameters
        rm_intra = self.param["rm_intra"]
        rm_inter = self.param['rm_inter']
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        if method == 'e': # force on element # 0.3 min
            # transfer array to gpu
            d_rm_intra = cupy.asanyarray(rm_intra*self.cell_size)
            d_xe = cupy.asanyarray(self.xe) # float32
            xe_F_intra = cupy.asanyarray(np.zeros_like(self.xe)) # zeros_like use the same dtype
            xe_F_inter = cupy.asanyarray(np.zeros_like(self.xe))
            d_ecid = cuda.to_device(self.ecid)
            d_alpha = cuda.to_device(alpha.astype(np.float32))
            d_beta = cuda.to_device(beta.astype(np.float32))
            d_gamma = cuda.to_device(gamma.astype(np.float32))

            # gpu thread number
            tpb = 64
            bpg = 128
            # computation
            cuda.synchronize()
            # force_gpu[bpg, tpb](d_xe, xe_F_intra, xe_F_inter, d_ecid, d_alpha, gamma, d_rm_intra, rm_inter)
            e_force_gpu[bpg, tpb](d_xe, xe_F_intra, xe_F_inter, d_ecid, d_alpha, d_beta, d_gamma, d_rm_intra, rm_inter)
            cuda.synchronize()
            self.xe_F_intra = xe_F_intra.get()
            self.xe_F_inter = xe_F_inter.get()
        elif method =='ee_cpu': # element-element force # 36 min
            row_intra, col_intra, xe_Fx_intra, xe_Fy_intra, row_inter, col_inter, xe_Fx_inter, xe_Fy_inter = ee_force_cpu(self.xe, self.ecid, alpha, beta, gamma, rm_intra*np.ones(self.ne,dtype=np.float32), rm_inter)
            s = (self.ne,self.ne)
            self.xe_Fx_intra = csr_matrix((xe_Fx_intra,(row_intra, col_intra)),shape=s)
            self.xe_Fy_intra = csr_matrix((xe_Fy_intra,(row_intra, col_intra)),shape=s)
            self.xe_Fx_inter = csr_matrix((xe_Fx_inter,(row_inter, col_inter)),shape=s)
            self.xe_Fy_inter = csr_matrix((xe_Fy_inter,(row_inter, col_inter)),shape=s)
        elif method == 'ee_gpu': # element-element force # 6s
            print('ee_gpu')
            pairs = find_pairs(self.xe,30) # 5.5s
            d_pairs = cuda.to_device(pairs)
            f_pairs = cuda.to_device(np.zeros_like(pairs).astype(np.float32))
            d_rm_intra = cuda.to_device(rm_intra*self.cell_size)
            d_xe = cuda.to_device(self.xe) # float32
            d_ecid = cuda.to_device(self.ecid)
            d_alpha = cuda.to_device(alpha.astype(np.float32))
            d_beta = cuda.to_device(beta.astype(np.float32))
            d_gamma = cuda.to_device(gamma.astype(np.float32))
            # gpu thread number
            tpb = 128
            bpg = 128
            # computation
            cuda.synchronize()
            ee_pairwise_forces_gpu[bpg, tpb](d_xe,d_pairs,f_pairs,d_ecid, d_alpha, d_beta, d_gamma, d_rm_intra, rm_inter)
            cuda.synchronize()
            self.pairs = pairs
            self.f_pairs = f_pairs.copy_to_host()

    # to-do: cell-cell force

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
            if self.alphashape_info['computed']:
                data['alpha_radius']= self.alpha_radius
            pickle.dump(data, f)
        print(f"saved as {filename}")

    def load_sim(self, sim_name: str , t: float, path: str = '.', rename: bool = True) -> None:
        '''
        Restore a simulation from `{path}/{sim_name}_{t}.pkl`

        Parameters
        ------
        sim_name : str
            Name of simulation
        t : float
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
            if 'alpha_radius' in data:
                print('.alpha_radius loaded')
                self.compute_alphashape(alpha=data['alpha_radius'])
        self.t = t
        self.update_xc()
        if rename:
            print(f'Simulation renamed as {sim_name}')
            self.sim_name = sim_name

class SEM1(CellBase):
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
        rd_ratio: float = 2,
        adata: Optional[AnnData] = None,
        cluster_key: str = 'leiden',
        spatial_key: str = 'spatial',
        xc: Optional[NDArray] = None,
        ctype: Optional[NDArray] = None,
        sim_name: str = 'untitled',# param: dict = {}
        seed: int = 1
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

        # adhesion based on gene simarity
        # if self.adata is None:
        #     self.corr_matrix = np.ones((self.nc,self.nc), dtype=np.float32)
        # else:
        #     X_em = self.adata.obsm[embedding_key]# default PCA matrix
        #     corr_matrix = np.corrcoef(X_em)
        #     c_min = 0.05
        #     corr_matrix[corr_matrix<c_min] = c_min
        #     self.corr_matrix = corr_matrix.astype(np.float32) # gene simarity matrix, (n_c*n_c)
            
        self.ns_default = 10 # alphashape default param
        
    def sim_gpu_test(self,rm_intra,rm_inter,alpha,beta,gamma,sigma,dt,T,enable_tqdm=True) -> None:
        # test pre defined parameters alpha,beta,sigma,gamma
        alpha = alpha.astype(np.float32)
        beta = beta.astype(np.float32)
        gamma = gamma.astype(np.float32)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.param["rm_intra"] = rm_intra
        self.param["rm_inter"] = rm_inter
        # d_rm_intra = cupy.asanyarray(rm_intra*self.cell_size)
        d_xe = cuda.to_device(self.xe)
        d_xe_F = cuda.to_device(self.xe)
        d_ecid = cuda.to_device(self.ecid)
        d_alpha = cuda.to_device(alpha)
        d_beta = cuda.to_device(beta)
        d_gamma = cuda.to_device(gamma)
        sigmadt = sqrt(dt) * sigma
        # gpu thread number
        tpb = 128
        bpg = 128#max((self.ne+tpb-1)//tpb,128)
        # print(tpb,bpg)
        # iteration
        cuda.synchronize()
        for t in tqdm(range(T),'Simulation',disable=not enable_tqdm):
            # a = np.sqrt((T-t)/T)
            # a = ((T-t)/T)**2
            a = 1
            # d_gamma = cuda.to_device(a*gamma)
            x_randt = cuda.to_device((sigmadt * a * self.rng.normal(0, 1, size=self.xe.shape)).astype(np.float32))#*self.cell_size[self.ecid,np.newaxis]#
            dynamics2d_gpu4_SEM1[bpg, tpb](d_xe, d_xe_F, d_ecid, d_alpha, d_beta, d_gamma, x_randt, rm_intra, rm_inter, dt)
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

    def sim_gpu_opt_test(self,rm_intra,rm_inter,alpha,beta,gamma,sigma,dt,T,enable_tqdm=True) -> None:
        # update: gpu thread number, pre-compute rm^6, call dynamics2d_gpu4_SEM1_opt()

        # test pre defined parameters alpha,beta,sigma,gamma
        alpha = alpha.astype(np.float32)
        beta = beta.astype(np.float32)
        gamma = gamma.astype(np.float32)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.param["rm_intra"] = rm_intra
        self.param["rm_inter"] = rm_inter
        rm6_intra = rm_intra**6
        rm6_inter = rm_inter**6
        # d_rm_intra = cupy.asanyarray(rm_intra*self.cell_size)
        d_xe = cuda.to_device(self.xe)
        d_xe_F = cuda.to_device(self.xe)
        d_ecid = cuda.to_device(self.ecid)
        d_alpha = cuda.to_device(alpha)
        d_beta = cuda.to_device(beta)
        d_gamma = cuda.to_device(gamma)
        sigmadt = sqrt(dt) * sigma
        # gpu thread number
        tpb = 128
        bpg = max((self.ne+tpb-1)//tpb,128)
        # print(tpb,bpg)
        # iteration
        cuda.synchronize()
        for t in tqdm(range(T),'Simulation',disable=not enable_tqdm):
            # a = np.sqrt((T-t)/T)
            # a = ((T-t)/T)**2
            a = 1
            # d_gamma = cuda.to_device(a*gamma)
            x_randt = cuda.to_device((sigmadt * a * self.rng.normal(0, 1, size=self.xe.shape)).astype(np.float32))#*self.cell_size[self.ecid,np.newaxis]#
            dynamics2d_gpu4_SEM1_opt[bpg, tpb](d_xe, d_xe_F, d_ecid, d_alpha, d_beta, d_gamma, x_randt, rm6_intra, rm6_inter, dt)
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

    def sim_gpu_test1(self,rm_intra,rm_inter,alpha,beta,gamma,sigma,dt,T,tocpu_interval = 10) -> None:
        import cupy 
        
        # test pre defined parameters alpha,beta,sigma,gamma
        alpha = alpha.astype(np.float32)
        beta = beta.astype(np.float32)
        gamma = gamma.astype(np.float32)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.param["rm_intra"] = rm_intra
        self.param["rm_inter"] = rm_inter
        self.xct = [[self.t,self.xc]]
        d_xe = cupy.asarray(self.xe)
        d_xe_F = cupy.asarray(self.xe)
        d_ecid = cupy.asarray(self.ecid)
        d_alpha = cupy.asarray(alpha)
        d_beta = cupy.asarray(beta)
        d_gamma = cupy.asarray(gamma)
        d_xc = cupy.asarray(self.xc)
        d_ne_per_cell = cupy.asarray(self.ne_per_cell, dtype = cupy.float32)
        sigmadt = sqrt(dt) * sigma
        # gpu thread number
        tpb = 128
        bpg = 128#max((self.ne+tpb-1)//tpb,128)
        # print(tpb,bpg)
        # iteration
        # cupy.random.seed(1)
        rng = cupy.random.RandomState(self.rng_seed)
        cuda.synchronize()
        for t in range(T):
            # a = np.sqrt((T-t)/T)
            # a = ((T-t)/T)**2
            a = 1
            x_randt = rng.randn(self.ne, 2, dtype=cupy.float32) * sigmadt * a#*self.cell_size[self.ecid,np.newaxis]#
            # x_randt = cupy.random.randn(self.ne, 2, dtype=cupy.float32) * sigmadt * a#*self.cell_size[self.ecid,np.newaxis]#
            # x_randt = cupy.asarray((sigmadt * a * self.rng.normal(0, 1, size=self.xe.shape)).astype(np.float32))#*self.cell_size[self.ecid,np.newaxis]#
            dynamics2d_gpu5_SEM1[bpg, tpb](d_xe, d_xe_F, d_xc, d_ecid, d_alpha, d_beta, d_gamma, x_randt, rm_intra, rm_inter, dt)
            cuda.synchronize()
            # var:t-1, var_F:t
            d_xe[:, :] = d_xe_F#d_xe, d_xe_F = d_xe_F, d_xe # swap is slower than copy
            cuda.synchronize()
            sum_x = cupy.bincount(d_ecid, weights=d_xe[:,0], minlength=self.nc)
            sum_y = cupy.bincount(d_ecid, weights=d_xe[:,1], minlength=self.nc)
            d_xc[:,0] = sum_x/d_ne_per_cell
            d_xc[:,1] = sum_y/d_ne_per_cell
            if t % tocpu_interval == 0:
                self.xct.append([self.t,d_xc.get()])
            #     print(self.t)
            self.t += 1
        # close
        cuda.synchronize()
        self.xe = d_xe.get()
        # self.xe = d_xe.copy_to_host()
        self.xc = d_xc.get()
        if t % tocpu_interval != 0:
            self.xct.append([self.t,self.xc])
        # self.update_xc()
        self.alphashape_info['computed'] = False # marks alpha shapes need to be updated
        
    def compute_force(self, method: str) -> None:
        """
        Compute inter/intracellular force
        
        add self.pairs[k] = [i,j]

        add self.f_pairs[k] stores force j -> i

        call `ee_pairwise_forces1_gpu_SEM1`

        Parameter
        ------
        method : str
            'e' for force on element;  'ee_gpu' for element-element force
        """
        # self.param = param
        # # get parameters
        # rm_intra = param["rm_intra"]*self.cell_size#1.5
        # rm_inter = param["rm_inter"]#3
        # gamma = param["gamma"]#0.01
        # alpha_max,alpha_min = param["alpha"]#0.25
        
        # cmax = self.corr_matrix.max()
        # cmin = self.corr_matrix.min()
        # if cmax==cmin:
        #     # corr_matrix is constant, set alpha to ones
        #     alpha = np.ones_like(self.corr_matrix)
        # else:
        #     # scale corr_matrix to [alpha_min,alpha_max]
        #     alpha = (alpha_max-alpha_min)/(cmax-cmin)*self.corr_matrix+(alpha_min*cmax-alpha_max*cmin)/(cmax-cmin)
        # if lr is not None:
        #     alpha+=lr

        # get parameters
        rm_intra = self.param["rm_intra"]
        rm_inter = self.param['rm_inter']
        # alpha = self.alpha
        # beta = self.beta
        # gamma = self.gamma
        # if method == 'e': # force on element # 0.3 min
        #     # transfer array to gpu
        #     # d_rm_intra = cupy.asanyarray(rm_intra*self.cell_size)
        #     d_xe = cupy.asanyarray(self.xe) # float32
        #     xe_F_intra = cupy.asanyarray(np.zeros_like(self.xe)) # zeros_like use the same dtype
        #     xe_F_inter = cupy.asanyarray(np.zeros_like(self.xe))
        #     d_ecid = cuda.to_device(self.ecid)
        #     d_alpha = cuda.to_device(alpha.astype(np.float32))
        #     d_beta = cuda.to_device(beta.astype(np.float32))
        #     d_gamma = cuda.to_device(gamma.astype(np.float32))

        #     # gpu thread number
        #     tpb = 64
        #     bpg = 128
        #     # computation
        #     cuda.synchronize()
        #     # force_gpu[bpg, tpb](d_xe, xe_F_intra, xe_F_inter, d_ecid, d_alpha, gamma, d_rm_intra, rm_inter)
        #     e_force_gpu[bpg, tpb](d_xe, xe_F_intra, xe_F_inter, d_ecid, d_alpha, d_beta, d_gamma, rm_intra, rm_inter)
        #     cuda.synchronize()
        #     self.xe_F_intra = xe_F_intra.get()
        #     self.xe_F_inter = xe_F_inter.get()
        # elif method =='ee_cpu': # element-element force # 36 min
        #     row_intra, col_intra, xe_Fx_intra, xe_Fy_intra, row_inter, col_inter, xe_Fx_inter, xe_Fy_inter = ee_force_cpu(self.xe, self.ecid, alpha, beta, gamma, rm_intra*np.ones(self.ne,dtype=np.float32), rm_inter)
        #     s = (self.ne,self.ne)
        #     self.xe_Fx_intra = csr_matrix((xe_Fx_intra,(row_intra, col_intra)),shape=s)
        #     self.xe_Fy_intra = csr_matrix((xe_Fy_intra,(row_intra, col_intra)),shape=s)
        #     self.xe_Fx_inter = csr_matrix((xe_Fx_inter,(row_inter, col_inter)),shape=s)
        #     self.xe_Fy_inter = csr_matrix((xe_Fy_inter,(row_inter, col_inter)),shape=s)
        if method == 'ee_gpu': # element-element force # 6s
            print('ee_gpu')
            pairs = find_pairs(self.xe,30) # 5.5s
            d_pairs = cuda.to_device(pairs)
            f_pairs = cuda.to_device(np.zeros_like(pairs).astype(np.float32))
            d_xe = cuda.to_device(self.xe) # float32
            d_ecid = cuda.to_device(self.ecid)
            d_alpha = cuda.to_device(self.alpha)
            d_beta = cuda.to_device(self.beta)
            # if np.any(gamma>0):
            #     print('set gamma = 0')
            # d_gamma = cuda.to_device(self.gamma*0)
            d_gamma = cuda.to_device(self.gamma)
            # gpu thread number
            tpb = 128
            bpg = 128
            # computation
            cuda.synchronize()
            ee_pairwise_forces1_gpu_SEM1[bpg, tpb](d_xe,d_pairs,f_pairs,d_ecid, d_alpha, d_beta, d_gamma, rm_intra, rm_inter)
            cuda.synchronize()
            self.pairs = pairs
            self.f_pairs = f_pairs.copy_to_host()
            self.compute_cell_forces() # cell-cell force abs symmetric
    
    def compute_force1(self):
        """
        Compute inter/intracellular force (element-element)
        
        add self.pairs[k] = [i,j]

        add self.f_pairs[k] stores force j -> i

        call `ee_pairwise_forces2_gpu_SEM1`, `ec_forces2_gpu_SEM1`
        """
        pairs = find_pairs(self.xe,30) # 5.5s
        d_pairs = cuda.to_device(pairs)
        f_pairs = cuda.to_device(np.zeros_like(pairs).astype(np.float32))
        d_xe = cuda.to_device(self.xe) # float32
        d_ecid = cuda.to_device(self.ecid)
        d_xc = cuda.to_device(self.xc)
        f_ec = cuda.to_device(np.zeros_like(self.xe).astype(np.float32))

        # get parameters
        rm_intra = self.param["rm_intra"]
        rm_inter = self.param['rm_inter']
        d_alpha = cuda.to_device(self.alpha)
        d_beta = cuda.to_device(self.beta)
        d_gamma = cuda.to_device(self.gamma)

        # gpu thread number
        tpb = 128
        bpg = 128

        # computation
        cuda.synchronize()
        ee_pairwise_forces2_gpu_SEM1[bpg, tpb](d_xe,d_pairs,f_pairs,d_ecid, d_alpha, d_beta, rm_intra, rm_inter)
        cuda.synchronize()
        ec_forces2_gpu_SEM1[bpg, tpb](d_xe,d_xc,f_ec,d_ecid,d_gamma)
        cuda.synchronize()
        self.pairs = pairs
        self.f_pairs = f_pairs.copy_to_host()
        self.f_ec = f_ec.copy_to_host()
        self.compute_cell_forces() # cell-cell force abs
        self.compute_contraction()

    def compute_cell_forces(self) -> None:
        """
        add self.f_cc: cell-cell force matrix (symmetric, abs)
        """
        f_abs = np.linalg.norm(self.f_pairs,axis=1)
        f_cc = cell_cell_forces(self.pairs,f_abs,self.ecid,self.nc)
        f_cc = csr_matrix(f_cc)
        I = f_cc.data<1e-1
        f_cc.data[I] = 0
        f_cc.eliminate_zeros()
        self.f_cc = f_cc
        if self.adata is not None:
            print("add .obsp['f_cc'] to adata")
            self.adata.obsp['f_cc'] = f_cc
    
    def compute_contraction(self):
        f_ec_abs = np.linalg.norm(self.f_ec,axis=1)
        self.f_c = np.zeros(self.nc)
        for i in range(self.ne):
            cid = self.ecid[i]
            self.f_c[cid] += f_ec_abs[i]

    def f_pairs_tocsr(self) -> Tuple[csr_array,csr_array]:
        """
        fx_csr[i, j] = fx_{j->i}

        row index i : under force

        column index j : exert force
        """
        # 行=受力粒子, 列=施力粒子
        # todo: use amplitude, angle to store forces instead of fx, fy
        i_idx = np.concatenate([self.pairs[:,0], self.pairs[:,1]])
        j_idx = np.concatenate([self.pairs[:,1], self.pairs[:,0]])
        fx_vals = np.concatenate([self.f_pairs[:,0], -self.f_pairs[:,0]])
        fy_vals = np.concatenate([self.f_pairs[:,1], -self.f_pairs[:,1]])
        fx_coo = coo_matrix((fx_vals, (i_idx, j_idx)), shape=(self.ne, self.ne))
        fy_coo = coo_matrix((fy_vals, (i_idx, j_idx)), shape=(self.ne, self.ne))
        fx_csr = fx_coo.tocsr()
        fy_csr = fy_coo.tocsr()
        if not (np.array_equal(fx_csr.indices, fy_csr.indices) and np.array_equal(fx_csr.indptr, fy_csr.indptr)):
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
            if 'pairs' in data:
                print('.pairs loaded\n.f_pairs loaded')
                self.pairs = data['pairs']
                self.f_pairs = data['f_pairs']
                self.compute_cell_forces()
            
        self.t = t
        self.update_xc()
        if rename:
            print(f'Simulation renamed as {sim_name}')
            self.sim_name = sim_name

@cuda.jit  
def force_gpu(xe, xe_F_intra, xe_F_inter, ecid, alpha, gamma, rm_intra, rm_inter):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    ne = xe.shape[0]
    for i in range(start, ne, stride):
        cid = ecid[i]
        for j in range(ne):
            if j == i :
                continue
            deltax = xe[i, 0]-xe[j, 0]
            deltay = xe[i, 1]-xe[j, 1]
            if abs(deltax) < 30 and abs(deltay) < 30:
                r = sqrt(deltax**2 + deltay**2)
                if ecid[j] == cid:
                    dV_intra = max(2*d_potential_LJ_gpu(r, rm_intra[cid], 1.5)+ gamma*r, -10.0)
                    xe_F_intra[i, 0] += -dV_intra * deltax
                    xe_F_intra[i, 1] += -dV_intra * deltay
                else:
                    dV_inter = max(alpha[cid,ecid[j]]*d_potential_LJ_gpu(r, rm_inter, 1.5), -10.0)
                    xe_F_inter[i, 0] += -dV_inter * deltax
                    xe_F_inter[i, 1] += -dV_inter * deltay

@cuda.jit
def dynamics2d_gpu2(xe, xe_F, ecid, alpha, gamma, x_randt, rm_intra, rm_inter, dt):
    """
    Simulation function
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    ne = xe.shape[0]
    for i in range(start, ne, stride):
        cid = ecid[i]
        for j in range(ne):
            if j == i :
                continue
            deltax = xe[i, 0]-xe[j, 0]
            deltay = xe[i, 1]-xe[j, 1]
            if abs(deltax) < 30 and abs(deltay) < 30:
                r = sqrt(deltax**2 + deltay**2)
                if ecid[j] == cid:
                    dV = max(2*d_potential_LJ_gpu(r, rm_intra[cid], 1.5)+ gamma*r, -10.0 )#
                else:
                    dV = max(alpha[cid,ecid[j]]*d_potential_LJ_gpu(r, rm_inter, 1.5), -10.0)
                xe_F[i, 0] += -dt * dV * deltax
                xe_F[i, 1] += -dt * dV * deltay
        xe_F[i, 0] += x_randt[i, 0]
        xe_F[i, 1] += x_randt[i, 1]

@cuda.jit
def dynamics2d_gpu3(xe, xe_F, ecid, alpha, beta, gamma, x_randt, rm_intra, rm_inter, dt):
    # add beta_i for each cell
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    ne = xe.shape[0]
    for i in range(start, ne, stride):
        cid = ecid[i]
        for j in range(ne):
            if j == i :
                continue
            deltax = xe[i, 0]-xe[j, 0]
            deltay = xe[i, 1]-xe[j, 1]
            if abs(deltax) < 30 and abs(deltay) < 30:
                r = sqrt(deltax**2 + deltay**2)
                if ecid[j] == cid:
                    dV = max(beta[cid]*d_potential_LJ_gpu(r, rm_intra[cid], 1.5) + gamma[cid]*r, -10.0 )
                else:
                    dV = max(alpha[cid,ecid[j]]*d_potential_LJ_gpu(r, rm_inter, 1.5), -10.0)
                xe_F[i, 0] += -dt * dV * deltax
                xe_F[i, 1] += -dt * dV * deltay
        xe_F[i, 0] += x_randt[i, 0]
        xe_F[i, 1] += x_randt[i, 1]

@cuda.jit
def dynamics2d_gpu3_SEM1(xe, xe_F, ecid, alpha, beta, gamma, x_randt, rm_intra, rm_inter, dt):
    # dynamics2d_gpu2
    # add beta_i for each cell
    # replace rm_intra[cid] by rm_intra
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    ne = xe.shape[0]
    for i in range(start, ne, stride):
        cid = ecid[i]
        for j in range(ne):
            if j == i :
                continue
            deltax = xe[i, 0]-xe[j, 0]
            deltay = xe[i, 1]-xe[j, 1]
            if abs(deltax) < 30 and abs(deltay) < 30:
                r = sqrt(deltax**2 + deltay**2)
                if ecid[j] == cid:
                    dV = max(beta[cid]*d_potential_LJ_gpu(r, rm_intra, 1.5) + gamma[cid]*r, -10.0 )
                else:
                    dV = max(alpha[cid,ecid[j]]*d_potential_LJ_gpu(r, rm_inter, 1.5), -10.0)
                xe_F[i, 0] += -dt * dV * deltax
                xe_F[i, 1] += -dt * dV * deltay
        xe_F[i, 0] += x_randt[i, 0]
        xe_F[i, 1] += x_randt[i, 1]

@cuda.jit
def dynamics2d_gpu4_SEM1(xe, xe_F, ecid, alpha, beta, gamma, x_randt, rm_intra, rm_inter, dt):
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
        cid = ecid[i]
        for j in range(ne):
            if j == i :
                continue
            deltax = xe[i, 0]-xe[j, 0]
            deltay = xe[i, 1]-xe[j, 1]
            if abs(deltax) < 30 and abs(deltay) < 30:
                r = sqrt(deltax**2 + deltay**2)
                if ecid[j] == cid:
                    dV = max(beta[cid]*d_potential_LJ_gpu(r, rm_intra, 1.5) + gamma[cid]/r**3, -10.0 )
                else:
                    dV = max(alpha[cid,ecid[j]]*d_potential_LJ_gpu(r, rm_inter, 1.5), -10.0)
                xe_F[i, 0] += -dt * dV * deltax
                xe_F[i, 1] += -dt * dV * deltay
        xe_F[i, 0] += x_randt[i, 0]
        xe_F[i, 1] += x_randt[i, 1]

@cuda.jit
def dynamics2d_gpu4_SEM1_opt(xe, xe_F, ecid, alpha, beta, gamma, x_randt, rm6_intra, rm6_inter, dt):
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
        # gamma_i = gamma[cid]
        fx = 0.0
        fy = 0.0
        for j in range(ne):
            if j == i :
                continue
            xj = xe[j, 0]
            yj = xe[j, 1]
            deltax = xi - xj
            deltay = yi - yj
            deltax_s = deltax*deltax
            deltay_s = deltay*deltay
            if deltax_s < 900 and deltay_s < 900:
                r2 = deltax_s + deltay_s
                cid_j = ecid[j]
                if cid_j == cid:
                    dV = beta_i*d_potential_LJ_gpu_r2(r2, rm6_intra, 1.5)# + gamma_i/r**3
                else:
                    dV = alpha[cid,cid_j]*d_potential_LJ_gpu_r2(r2, rm6_inter, 1.5)

                if dV < -10.0:
                    dV = -10.0
        ## 每次迭代都乘以 dt
        #         xe_F[i, 0] += -dt * dV * deltax
        #         xe_F[i, 1] += -dt * dV * deltay
        # xe_F[i, 0] += x_randt[i, 0]
        # xe_F[i, 1] += x_randt[i, 1]
        ## 最后统一乘以 dt，浮点运算不等价
                fx -= dV * deltax
                fy -= dV * deltay
        xe_F[i, 0] += dt * fx + x_randt[i, 0]
        xe_F[i, 1] += dt * fy + x_randt[i, 1]
        ##

@cuda.jit
def dynamics2d_gpu5_SEM1(xe, xe_F, xc, ecid, alpha, beta, gamma, x_randt, rm_intra, rm_inter, dt):
    """
    dynamics2d_gpu2

    add beta_i for each cell

    inter dV = max(beta[cid]*d_potential_LJ_gpu(r, rm_intra, 1.5), -10.0 )

    intra dV = max(alpha[cid,ecid[j]]*d_potential_LJ_gpu(r, rm_inter, 1.5), -10.0)

    xe_F[i, 0] += x_randt[i, 0] + gamma[cid]*dt*deltax/r
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    ne = xe.shape[0]
    for i in range(start, ne, stride):
        cid = ecid[i]
        for j in range(ne):
            if j == i :
                continue
            deltax = xe[i, 0]-xe[j, 0]
            deltay = xe[i, 1]-xe[j, 1]
            if abs(deltax) < 30 and abs(deltay) < 30:
                r = sqrt(deltax**2 + deltay**2)
                if ecid[j] == cid:
                    dV = max(beta[cid]*d_potential_LJ_gpu(r, rm_intra, 1.5), -10.0 )
                else:
                    dV = max(alpha[cid,ecid[j]]*d_potential_LJ_gpu(r, rm_inter, 1.5), -10.0)
                xe_F[i, 0] += -dt * dV * deltax
                xe_F[i, 1] += -dt * dV * deltay
        deltax = xe[i, 0]-xc[cid,0]
        deltay = xe[i, 1]-xc[cid,1]
        r = sqrt(deltax**2 + deltay**2)
        # constant repulsion force between xe and xc
        xe_F[i, 0] += x_randt[i, 0] + gamma[cid]*dt*deltax
        xe_F[i, 1] += x_randt[i, 1] + gamma[cid]*dt*deltay

@cuda.jit  
def e_force_gpu(xe, xe_F_intra, xe_F_inter, ecid, alpha, beta, gamma, rm_intra, rm_inter):
    # dynamics2d_gpu3
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    ne = xe.shape[0]
    for i in range(start, ne, stride):
        cid = ecid[i]
        for j in range(ne):
            if j == i :
                continue
            deltax = xe[i, 0]-xe[j, 0]
            deltay = xe[i, 1]-xe[j, 1]
            if abs(deltax) < 30 and abs(deltay) < 30:
                r = sqrt(deltax**2 + deltay**2)
                if ecid[j] == cid:
                    dV_intra = max(beta[cid]*d_potential_LJ_gpu(r, rm_intra[cid], 1.5) + gamma[cid]*r, -10.0 )
                    xe_F_intra[i, 0] += -dV_intra * deltax
                    xe_F_intra[i, 1] += -dV_intra * deltay
                else:
                    dV_inter = max(alpha[cid,ecid[j]]*d_potential_LJ_gpu(r, rm_inter, 1.5), -10.0)
                    xe_F_inter[i, 0] += -dV_inter * deltax
                    xe_F_inter[i, 1] += -dV_inter * deltay

@cuda.jit
def ee_pairwise_forces_gpu(xe, pairs, pair_force, ecid, alpha, beta, gamma, rm_intra, rm_inter):
    """
    对预计算得到的每个 element pair 计算相互作用力，结果存储在 pair_force 数组中，
    每行保存 (fx, fy)。
    
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
    np = pairs.shape[0]
    for k in range(start, np, stride):
        i = pairs[k, 0]
        j = pairs[k, 1]
        deltax = xe[i, 0]-xe[j, 0]
        deltay = xe[i, 1]-xe[j, 1]
        r = sqrt(deltax**2 + deltay**2)
        cid_i = ecid[i]
        cid_j = ecid[j]
        if ecid[j] == cid_i:
            dV = beta[cid_i] * d_potential_LJ_gpu(r, rm_intra[cid_i], 1.5) + gamma[cid_i] * r
        else:
            dV = alpha[cid_i, cid_j] * d_potential_LJ_gpu(r, rm_inter, 1.5)
        if dV < -10.0:
            dV = -10.0
        pair_force[k, 0] = -dV * deltax
        pair_force[k, 1] = -dV * deltay

@cuda.jit
def ee_pairwise_forces_gpu_SEM1(xe, pairs, pair_force, ecid, alpha, beta, gamma, rm_intra, rm_inter):
    # ee_pairwise_forces_gpu
    # replace rm_intra[cid_i] by rm_intra
    """
    对预计算得到的每个 element pair 计算相互作用力，结果存储在 pair_force 数组中，
    每行保存 (fx, fy)。
    
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
    np = pairs.shape[0]
    for k in range(start, np, stride):
        i = pairs[k, 0]
        j = pairs[k, 1]
        deltax = xe[i, 0]-xe[j, 0]
        deltay = xe[i, 1]-xe[j, 1]
        r = sqrt(deltax**2 + deltay**2)
        cid_i = ecid[i]
        cid_j = ecid[j]
        if ecid[j] == cid_i:
            dV = beta[cid_i] * d_potential_LJ_gpu(r, rm_intra, 1.5) + gamma[cid_i] * r
        else:
            dV = alpha[cid_i, cid_j] * d_potential_LJ_gpu(r, rm_inter, 1.5)
        if dV < -10.0:
            dV = -10.0
        pair_force[k, 0] = -dV * deltax
        pair_force[k, 1] = -dV * deltay

@cuda.jit
def ee_pairwise_forces1_gpu_SEM1(xe, pairs, pair_force, ecid, alpha, beta, gamma, rm_intra, rm_inter):
    """
    dynamics2d_gpu4_SEM1

    inter dV = beta[cid_i] * d_potential_LJ_gpu(r, rm_intra, 1.5) + gamma[cid_i]/r**3

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
    np = pairs.shape[0]
    for k in range(start, np, stride):
        i = pairs[k, 0]
        j = pairs[k, 1]
        deltax = xe[i, 0]-xe[j, 0]
        deltay = xe[i, 1]-xe[j, 1]
        r = sqrt(deltax**2 + deltay**2)
        cid_i = ecid[i]
        cid_j = ecid[j]
        if ecid[j] == cid_i:
            dV = beta[cid_i] * d_potential_LJ_gpu(r, rm_intra, 1.5) + gamma[cid_i]/r**3
        else:
            dV = alpha[cid_i, cid_j] * d_potential_LJ_gpu(r, rm_inter, 1.5)
        if dV < -10.0:
            dV = -10.0
        pair_force[k, 0] = -dV * deltax
        pair_force[k, 1] = -dV * deltay

# @cuda.jit
# useless because find_pairs() incorprates the distance threshold
# def ee_pairwise_forces1_1_gpu_SEM1(xe, pairs, pair_force, ecid, alpha, beta, gamma, rm_intra, rm_inter):
#     """
#     dynamics2d_gpu4_SEM1

#     inter dV = beta[cid_i] * d_potential_LJ_gpu(r, rm_intra, 1.5) + gamma[cid_i]/r**3

#     intra dV = alpha[cid_i, cid_j] * d_potential_LJ_gpu(r, rm_inter, 1.5)

#     对预计算得到的每个 element pair 计算相互作用力，结果存储在 pair_force 数组中，
#     每行保存 (fx, fy)。
    
#     Parameters
#     ------
#       xe         : (ne, 2) 的 element 坐标数组
#       pairs      : (num_pairs, 2) 的预计算 pair 索引数组，每行 (i, j)
#       pair_force : (num_pairs, 2) 的输出数组，存储每个 pair j->i的力
#       ecid       : (ne,) 数组，记录每个 element 属于哪个细胞
#       alpha      : (nc, nc) 数组，不同细胞间的相互作用参数
#       beta       : (nc,) 数组，同一细胞内相互作用参数
#       gamma      : (nc,) 数组，同一细胞内额外的刚性参数
#       rm_intra   : (nc,) 数组，每个细胞内部的 rm 参数 (通过细胞ID索引)
#       rm_inter   : 标量，不同细胞间的 rm 参数
#     """
#     start = cuda.grid(1)
#     stride = cuda.gridsize(1)
#     np = pairs.shape[0]
#     for k in range(start, np, stride):
#         i = pairs[k, 0]
#         j = pairs[k, 1]
#         deltax = xe[i, 0]-xe[j, 0]
#         deltay = xe[i, 1]-xe[j, 1]
#         if abs(deltax) < 30 and abs(deltay) < 30:
#             r = sqrt(deltax**2 + deltay**2)
#             cid_i = ecid[i]
#             cid_j = ecid[j]
#             if ecid[j] == cid_i:
#                 dV = beta[cid_i] * d_potential_LJ_gpu(r, rm_intra, 1.5) + gamma[cid_i]/r**3
#             else:
#                 dV = alpha[cid_i, cid_j] * d_potential_LJ_gpu(r, rm_inter, 1.5)
#             if dV < -10.0:
#                 dV = -10.0
#             pair_force[k, 0] = -dV * deltax
#             pair_force[k, 1] = -dV * deltay
    
@cuda.jit
def ee_pairwise_forces2_gpu_SEM1(xe, pairs, pair_force, ecid, alpha, beta, rm_intra, rm_inter):
    # dynamics2d_gpu5_SEM1
    # ee_pairwise_forces_gpu
    # replace rm_intra[cid_i] by rm_intra
    """
    对预计算得到的每个 element pair 计算相互作用力，结果存储在 pair_force 数组中，
    每行保存 (fx, fy)。
    
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
        deltax = xe[i, 0]-xe[j, 0]
        deltay = xe[i, 1]-xe[j, 1]
        r = sqrt(deltax**2 + deltay**2)
        cid_i = ecid[i]
        cid_j = ecid[j]
        if ecid[j] == cid_i:
            dV = beta[cid_i] * d_potential_LJ_gpu(r, rm_intra, 1.5)
        else:
            dV = alpha[cid_i, cid_j] * d_potential_LJ_gpu(r, rm_inter, 1.5)
        if dV < -10.0:
            dV = -10.0
        pair_force[k, 0] = -dV * deltax
        pair_force[k, 1] = -dV * deltay

@cuda.jit 
def ec_forces2_gpu_SEM1(xe,xc,force,ecid,gamma):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    ne = xe.shape[0]
    for i in range(start, ne, stride):
        cid = ecid[i]
        deltax = xe[i, 0]-xc[cid,0]
        deltay = xe[i, 1]-xc[cid,1]
        r = sqrt(deltax**2 + deltay**2)
        force[i,0] = gamma[cid]*deltax
        force[i,1] = gamma[cid]*deltay
    
def ee_force_cpu(xe, ecid, alpha, beta, gamma, rm_intra, rm_inter):
    # dynamics2d_gpu3
    xe_Fx_intra = []
    xe_Fy_intra = []
    xe_Fx_inter = []
    xe_Fy_inter = []
    row_intra = []
    col_intra = []
    row_inter = []
    col_inter = []
    ne = xe.shape[0]
    for i in range(ne):
        cid = ecid[i]
        for j in range(i+1,ne):
            # if j == i :
            #     continue
            deltax = xe[i, 0]-xe[j, 0]
            deltay = xe[i, 1]-xe[j, 1]
            if abs(deltax) < 30 and abs(deltay) < 30:
                r = sqrt(deltax**2 + deltay**2)
                if ecid[j] == cid:
                    dV_intra = max(beta[cid]*d_potential_LJ_cpu(r, rm_intra[cid], 1.5) + gamma[cid]*r, -10.0 )
                    xe_Fx_intra.append(-dV_intra * deltax) 
                    xe_Fy_intra.append(-dV_intra * deltay)
                    row_intra.append(i)
                    col_intra.append(j)
                else:
                    dV_inter = max(alpha[cid,ecid[j]]*d_potential_LJ_cpu(r, rm_inter, 1.5), -10.0)
                    xe_Fx_inter.append(-dV_inter * deltax)
                    xe_Fy_inter.append(-dV_inter * deltay)
                    row_inter.append(i)
                    col_inter.append(j)
    return row_intra, col_intra, xe_Fx_intra, xe_Fy_intra, row_inter, col_inter, xe_Fx_inter, xe_Fy_inter

class cellshape_GT(CellBase):
    """Cell shape representation for experimental data visualization"""
    def __init__(self, 
                 xe: NDArray,
                 ecid: NDArray,
                 ceidn: NDArray,
                 xc: Optional[NDArray] = None,
                 ctype: Optional[NDArray] = None,
                 color_list: Optional[NDArray] = None,
                 adata: Optional[AnnData] = None,
                 spatial_key: str = 'spatial',
                 cluster_key: str = 'leiden'):
        self.nc = ceidn.shape[0]-1
        super().__init__(adata, xc, ctype, cluster_key, spatial_key, color_list)
        # Visualization-specific properties
        self.xe = xe
        self.ecid = ecid
        self.ceidn = ceidn
        self.dim = xe.shape[1]
        self.ne_per_cell: NDArray = ceidn[1:]-ceidn[:-1]
        if xc is None and adata is None:
            print('compute xc from xe')
            self.update_xc()
    
    def __repr__(self):
        return f'Cell Number: {self.nc}\nElement Number: {self.xe.shape[0]}\nDim: {self.dim}'