from typing import Optional, Tuple, Union, List, Dict
from numba import cuda, njit, prange
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree, Delaunay, distance
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
from anndata import AnnData
from .._utils import AlphaShape
from matplotlib.colors import to_rgb
import seaborn as sns
import warnings
from tqdm import tqdm

class CellBase:
    """Base class for cell morphology representations"""
    
    adata: Optional[AnnData]
    """Linked annotation data. None if initialized from direct inputs (xc/ctype)."""
    nc: int
    """cell number"""
    dim: int 
    """spatial dimension"""
    xc: NDArray
    """cell coordinates. shape=(nc,dim)"""
    ctype: NDArray
    """cell type code. shape=(nc)"""
    ctype_list: NDArray
    """cell type category. shape=(nct)"""
    color_list: NDArray
    """colors for cell type category. shape=(nct,3)"""
    ne: int
    """element number"""
    xe: NDArray
    """element coordinates. shape=(ne,dim)"""
    ecid: NDArray
    """id of cell to which each element belongs. shape: (ne)"""
    ceidn: NDArray
    """first elements id of each cell (indptr). shape=(nc+1)
    
    Elements of cell i can be retrieve by ceidn[i]:ceidn[i+1]. 
    
    Number of elements of cell i is ceidn[i+1]-ceidn[i]"""
    scale: float
    """normalization scaling factor"""
    deltax: NDArray
    """normalization scaling offset. shape=(1,dim)"""
    contact_matrix: Union[None, csr_matrix]
    """cell-cell contact matrix. shape=(nc,nc)"""
    spatial_distances: Dict[str, csr_matrix]
    """dictionary to store cell-cell distance matrix"""
    alphashape: List[AlphaShape]
    """list of alpha-shape object for each cell. len=nc"""
    alphashape_info: Dict[str, Union[bool, float, int, None]]
    """information of alpha-shape"""
    alpha_radius : NDArray
    """alpha radius of each cell"""
    ns_default: int
    """number of augment points for alpha-shape"""
    def __init__(self,
        adata: Optional[AnnData] = None,
        xc: Optional[NDArray] = None,
        ctype: Optional[NDArray] = None,
        cluster_key: str = 'leiden',
        spatial_key: str = 'spatial',
        color_list: Optional[NDArray] = None
    ):
        """ Initialize common attributes"""
        
        self.adata = adata
        self.contact_matrix = None
        self.spatial_distances  = dict()
        self.alphashape_info = {'computed': False, 'alpha': None, 'ns': None, 'r': None}
        self.ns_default = 0
        self.scale = 1.
        self.deltax = np.zeros(2)
        
        # Initialize common attributes
        if self.adata is not None:
            self._init_from_adata(cluster_key,spatial_key)
        else:
            self._init_from_direct_inputs(xc,ctype,color_list)

        # Set default colors if not provided
        if self.color_list is None:
            self.set_color()

        self.alpha_radius = np.zeros(self.nc)
        
    def _init_from_adata(self,cluster_key,spatial_key):
        """Initialize properties from AnnData object"""
        assert self.adata is not None
        if self.adata.is_view:
            warnings.warn("adata is a view. Consider using `adata.copy()`, otherwise sem.adata is not link to the input Anndata object", UserWarning)
            self.adata = self.adata.copy()
        if spatial_key in self.adata.obsm:
            self.xc = self.adata.obsm[spatial_key].astype(np.float32)
            self.nc, self.dim = self.xc.shape
        else:
            raise KeyError(f"Spatial key '{spatial_key}' not found in adata.obsm")
        
        if cluster_key in self.adata.obs:
            self.ctype = self.adata.obs[cluster_key].cat.codes.to_numpy()
            self.ctype_list = self.adata.obs[cluster_key].cat.categories.to_numpy()
        else:
            raise KeyError(f"Cluster key '{cluster_key}' not found in adata.obs")
        
        if f'{cluster_key}_colors' in self.adata.uns:
            color_list = self.adata.uns[f'{cluster_key}_colors']
            if isinstance(color_list[0], str):
                color_list = np.array([to_rgb(x) for x in color_list])
        else:
            color_list = None
        self.color_list = color_list

    def _init_from_direct_inputs(self,xc,ctype,color_list):
        """Initialize properties from direct inputs"""
        if xc is None:
            raise ValueError("Either adata or xc must be provided")
        else:
            self.xc = xc.astype(np.float32)
            self.nc, self.dim = xc.shape
        
        self.ctype = ctype
        if self.ctype is None:
            warnings.warn('ctype are not provided, set cell type to 0')
            self.ctype = np.zeros(self.nc, dtype=int)
        self.ctype_list = np.unique(self.ctype)

        self.color_list = color_list
        if self.color_list is not None:
            if isinstance(self.color_list[0], str):
                self.color_list = np.array([to_rgb(x) for x in self.color_list])

    def set_color(self, cmap_name: Optional[str] = None, n_colors: Optional[int] = None) -> None:
        """
        Set a color map for `sem.ctype` (cell types)

        Parameters
        -------
        cmap_name: str
            Name of a seaborn color palette (`seaborn.color_palette`)
        n_colors: int, optional
            Number of colors to generate. Defaults to the number of cell types.
        """
        n = n_colors if n_colors is not None else len(self.ctype_list)
        palette = sns.color_palette(cmap_name, n_colors=n)
        n_unique = len(set(palette))
        if n_unique < len(self.ctype_list):
            warnings.warn(
                f"Color palette '{cmap_name}' produced {n_unique} unique color(s), "
                f"but there are {len(self.ctype_list)} cell types. "
                "Some cell types will share colors.",
                UserWarning, stacklevel=2
            )
        self.color_list = np.array(palette)

    def compute_contact(self,
                        k: int = 8,
                        d_th: Optional[float] = None,
                        add_key: str = 'contacts') -> None:
        """
        Compute cell-cell contacts

        add .contact_matrix (csr_matrix)

        stored in adata.obsp[add_key]

        Parameters
        ------
        k : int, default=8
            Number of neighbors
        d_th : Optional[float], default=None
            Distance threshold
        add_key : str, default='contacts'
            Key for storing cell-cell contacts matrix to .obsp
        """
        if d_th is None:
            d_th = 2*self._get_e_radius() 
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(self.xe)
        distances, indices = nbrs.kneighbors(self.xe)
        row_indices = []
        col_indices = []
        data = []
        for ci in range(self.nc):
            neighbor = indices[self.ceidn[ci]:self.ceidn[ci+1], 1:].flatten()
            d_contact = distances[self.ceidn[ci]:self.ceidn[ci+1], 1:].flatten()
            contact_eid = np.unique(neighbor[d_contact<=d_th])
            contact_cid = self.ecid[contact_eid]
            for cj in contact_cid[contact_cid!=ci]:
                row_indices.append(ci)
                col_indices.append(cj)
                data.append(1)

        # make symmetrical
        contact_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(self.nc, self.nc))
        self.contact_matrix = (contact_matrix+contact_matrix.T)/2 # contact_matrix is csr_matrix. tocoo() do not have .coords
        
        # if linked with adata, add contact_matrix to .obsp[add_key]
        if self.adata is not None:
            print(f"add .obsp['{add_key}'], .uns['{add_key}'] to adata")
            self.adata.obsp[add_key] = self.contact_matrix
            self.adata.uns[add_key] = {'k':k, 'd_th':d_th}

    def _get_e_radius(self) -> float: # replaced in sub-class
        """get default d_th for computing contact and alphashape"""
        return 1.0
    
    def compute_distance(self,
                         method: str,
                         k: int = 3,
                         return_distances: bool = False) -> Union[csr_matrix, None]:
        """
        Compute cell-cell distances in contact matrix

        Add .spatial_distances[method] (csr_matrix)

        Parameters
        ------
        method : str
            valid methods: 'knn', 'delaunay', 'contact'
        k : int, default: 3
            k for knn
        return_distances : bool, default: False
            Whether to return the distances matrix
        
        Return
        ------
        distance_matrix : csr_matrix
            Cell-cell distances if return_distances is True, None otherwise
        """

        xc = self.xc
        distance_matrix = lil_matrix((self.nc, self.nc))
        assert method in ('knn', 'delaunay', 'contact'), f"method must be 'knn' or 'delaunay', got {method}"
        if method == 'knn':
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(xc)
            distances,indices = nbrs.kneighbors(xc)
            for i in range(self.nc):
                for nj,j in enumerate(indices[i,1:]):
                    distance_matrix[i, j] = distances[i,nj+1]
                    distance_matrix[j, i] = distances[i,nj+1]
        
        if method == 'delaunay':
            tri = Delaunay(xc)
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        d = distance.euclidean(xc[simplex[i]],xc[simplex[j]])
                        distance_matrix[simplex[i], simplex[j]] = d
                        distance_matrix[simplex[j], simplex[i]] = d

        if method == 'contact':
            if self.contact_matrix is None:
                raise ValueError('contact is not computed')
            else:
                indices = self.contact_matrix.indices
                indptr = self.contact_matrix.indptr
                for i in range(self.nc):
                    for j in indices[indptr[i]:indptr[i+1]]:
                        d = distance.euclidean(xc[i],xc[j])
                        distance_matrix[i, j] = d
                        distance_matrix[j, i] = d
        if return_distances:
            return distance_matrix.tocsr()
        else:
            self.spatial_distances[method] = distance_matrix.tocsr()
    
    def compute_alphashape(
            self,
            alpha: Optional[Union[NDArray,float]] = None,
            ns: Optional[int] = None,
            r: Optional[float] = None
    ) -> None:
        """
        Compute alpha-shape for each cell
        
        Add .alphashape: List[AlphaShape]
        """
        if r is None:
            r = self._get_e_radius()/2
        if ns is None:
            ns = self.ns_default
        alpha_array = np.full(self.nc, alpha) if isinstance(alpha, float) else alpha
        # Check if recomputing alphashape are needed
        if not self.alphashape_info['computed'] or self.alphashape_info['ns']!=ns or self.alphashape_info['r']!=r:
            print(f"Computing alpha-shape with parameters: alpha={alpha}, ns={ns}, r={r}")
            xe = self.xe*self.scale+self.deltax
            self.alphashape = []
            for cid in tqdm(range(self.nc),'Processing Cell Shapes'):
                alpha_i = alpha_array[cid] if alpha_array is not None else None
                shp = AlphaShape(xe[self.ceidn[cid]:self.ceidn[cid+1]],alpha=alpha_i,ns=ns,r=r*self.scale)
                if alpha_i is None:
                    shp.update(1.5*shp.alpha_best)
                    shp.close_hole()
                self.alpha_radius[cid] = shp.alpha
                self.alphashape.append(shp)
        elif self.alphashape_info['alpha']!=alpha:
            print(f"Updating alpha to {alpha}")
            for cid,shp in enumerate(self.alphashape):
                alpha_i = alpha_array[cid] if alpha_array is not None else None
                if alpha_i is None:
                    if shp.alpha_best is None:
                        shp.optimize_alpha()
                    alpha_i = 1.5*shp.alpha_best
                shp.update(alpha_i)
                self.alpha_radius[cid] = shp.alpha
        # else do nothing

        # update alphashape_info
        self.alphashape_info['computed'] = True
        self.alphashape_info.update({'alpha':alpha, 'ns':ns, 'r':r})

    def get_alpha(self) -> Union[NDArray, None]:
        """
        Get alpha radius of each cell

        Return
        ------
        alpha : Union[np.ndarray, None]
            Alpha radius of each cell if alphashapes have been computed, otherwise None
        """
        # if self.alphashape_info['computed']:
            # alpha = np.zeros(self.nc)
            # for i in range(self.nc):
            #     alpha[i] = self.alphashape[i].alpha
        #     return 
        # else:
        if not self.alphashape_info['computed']:
            warnings.warn('alphashape is not computed')
        return self.alpha_radius
    
    def get_area(self) -> Union[NDArray, None]:
        """
        Get cell areas

        Return
        ------
        area : Union[np.ndarray, None]
            Cell areas if alphashapes have been computed, otherwise None
        """
        if self.alphashape_info['computed']:
            area = np.zeros(self.nc)
            for i in range(self.nc):
                area[i] = self.alphashape[i].get_area()
            return area
        else:
            warnings.warn('alphashape is not computed')
            return None
    
    def get_elements(self, i: int, scaling: bool = True,) -> NDArray:
        '''
        Get elements of cell i

        Parameters
        ------
        i : int 
            Cell index
        scaling: bool

        Return 
        ------
        xe : np.ndarray 
            Element coordinates of cell i
        '''
        xe = self.xe[self.ceidn[i]:self.ceidn[i+1]]
        return xe*self.scale+self.deltax if scaling else xe
    
    def get_box(self, cid_list, scaling: bool = True, rotation: Optional[Dict]= None) -> NDArray:
        xe = []
        for cid in cid_list:
            xe.append(self.get_elements(cid, scaling))
        xe = np.concatenate(xe)
        if rotation is not None:
            center = rotation['center']
            theta = rotation['theta']
            RT = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).T
            xe = (xe-center) @ RT + center
        xmax = xe.max(axis=0)
        xmin = xe.min(axis=0)
        return np.hstack((xmin,xmax))

    def get_neighbors(self, center_i: int):
        indices = self.contact_matrix.indices
        indptr = self.contact_matrix.indptr
        contact_i = indices[indptr[center_i]:indptr[center_i+1]]
        return contact_i
    
    def update_xc(self) -> None:
        """
        Update .xc (cell coordinates)
        """
        self.xc = np.array([np.mean(self.xe[self.ceidn[i]:self.ceidn[i+1]], axis=0) for i in range(self.nc)])

    def select_cell(self,box) -> NDArray:
        """
        select cell

        Parameter
        ------
        box : [xl,yl,xu,yu]

        Return 
        ------
        cid_list : np.ndarray
            array of selected cells' index 
        """
        xl, xr = np.sort([box[0],box[2]])
        yl, yr = np.sort([box[1],box[3]])
        spatial_coor = self.xc*self.scale+self.deltax
        return np.where((spatial_coor[:,0]>xl) & (spatial_coor[:,0]<xr) & (spatial_coor[:,1]>yl) & (spatial_coor[:,1]<yr))[0]


def d_potential_LJ_cpu(r, rm, epsilon):
    rs6 = (rm/r)**6
    return epsilon*r**-2*(rs6-rs6*rs6)

@cuda.jit(device=True)
def d_potential_LJ_gpu(r, rm, epsilon):
    rs6 = (rm/r)**6
    return epsilon*r**-2*(rs6-rs6*rs6)

@cuda.jit(device=True)
def d_potential_LJ_gpu_r2(r2, rm6, epsilon):
    rs6 = rm6/r2**3
    return epsilon*r2**-1*(rs6-rs6*rs6)

@cuda.jit(device=True)
def potential_LJ_gpu(r, rm, epsilon):
    rs6 = (rm/r)**6
    return epsilon*(rs6**2-2*rs6)
    # pass

@njit()
def cell_cell_forces(pairs, f_abs, ecid, nc):
    """
    pairs:  (m,2) 整型, 粒子对索引
    f_abs:  (m,)   浮点, 每对的绝对力值
    ecid:   (ne,)  整型, element->cell 映射
    nc:     int    cell 数量
    返回:
      f_cc: (nc,nc) 浮点, cell-cell 力矩阵
    """
    # 初始化
    f_cc = np.zeros((nc, nc), dtype=f_abs.dtype)
    m = pairs.shape[0]
    for k in range(m):
        e1 = pairs[k, 0]
        e2 = pairs[k, 1]
        c1 = ecid[e1]
        c2 = ecid[e2]
        if c1 != c2:# 保证对角线为0
            v = f_abs[k]
            # 累加到 (c1,c2) 与 (c2,c1)
            f_cc[c1, c2] += v
            f_cc[c2, c1] += v
    return f_cc

def find_pairs(xe,thr,metric='chebyshev'):
    """
    找出给定点集中距离小于阈值的所有点对
    
    参数：
    points : numpy.ndarray
        点的坐标数组，形状为(N, D), N为点数, D为维度
    threshold : float
        距离阈值
    metric : str, 可选
        距离度量方式，支持：'euclidean'(默认), 'manhattan', 'chebyshev'
    
    返回：
    numpy.ndarray
        符合条件的点对数组，形状为(n, 2)，每行包含两个点的索引
    """
    # 将度量名称转换为Minkowski参数
    metric_params = {
        'euclidean': 2,
        'manhattan': 1,
        'chebyshev': np.inf
    }
    tree = cKDTree(xe)
    pairs = tree.query_pairs(thr, p=metric_params[metric], output_type='ndarray')
    del tree
    return pairs

@njit(parallel=True)
def virial_stess_tensor(indptr, indices, fx, fy, xe):
    """
    indptr  : CSR  indptr
    indices : CSR  indices
    fx, fy  : CSR 力分量 data
    xe      : (ne,2) 的坐标数组
    returns : (ne,2,2) 的应力张量数组
    """
    ne = xe.shape[0]
    e_sigma = np.zeros((ne,2,2), dtype=xe.dtype)
    for i in prange(ne):
        # 累加局部四个分量
        s_xx = 0.0
        s_xy = 0.0
        s_yx = 0.0
        s_yy = 0.0
        # 遍历第 i 行的所有非零项 行=受力粒子
        start = indptr[i]
        end   = indptr[i+1]
        xi, yi = xe[i,0], xe[i,1]
        for ptr in range(start, end):
            j   = indices[ptr]
            dx  = xe[j,0] - xi
            dy  = xe[j,1] - yi
            fxi = fx[ptr]
            fyi = fy[ptr]
            # σ_αβ += r_{ij,α} * f_{ji,β} / 2
            s_xx += dx * fxi * 0.5
            s_xy += dx * fyi * 0.5
            s_yx += dy * fxi * 0.5
            s_yy += dy * fyi * 0.5
        e_sigma[i,0,0] = s_xx
        e_sigma[i,0,1] = s_xy
        e_sigma[i,1,0] = s_yx
        e_sigma[i,1,1] = s_yy
    return e_sigma