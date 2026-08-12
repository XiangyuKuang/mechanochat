from typing import Optional, Union, TYPE_CHECKING, List, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors
import warnings

def compute_distance(
        xc: NDArray,
        method: str,
        k: Optional[int] = 3,
        contact_matrix: Optional[csr_matrix] = None,
) -> csr_matrix:
    """
    Compute cell-cell distances

    Parameters
    ------
    method : str
        valid methods: 'knn', 'delaunay'
    k : int, default: 3
        k for knn
    
    Return
    ------
    distance_matrix : csr_matrix
    """

    # xc = self.xc
    nc = xc.shape[0]
    distance_matrix = lil_matrix((nc, nc))
    if contact_matrix is None:
        assert method in ('knn', 'delaunay'), f"method must be 'knn' or 'delaunay', got {method}"
        if method == 'knn':
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(xc)
            distances,indices = nbrs.kneighbors(xc)
            for i in range(nc):
                for nj,j in enumerate(indices[i,1:]):
                    distance_matrix[i, j] = distances[i,nj+1]
                    distance_matrix[j, i] = distances[i,nj+1]
        
        if method == 'delaunay':
            tri = Delaunay(xc)
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        d = euclidean(xc[simplex[i]],xc[simplex[j]])
                        distance_matrix[simplex[i], simplex[j]] = d
                        distance_matrix[simplex[j], simplex[i]] = d
    else:
        indices = contact_matrix.indices
        indptr = contact_matrix.indptr
        for i in range(nc):
            for j in indices[indptr[i]:indptr[i+1]]:
                d = euclidean(xc[i],xc[j])
                distance_matrix[i, j] = d
                distance_matrix[j, i] = d
    return distance_matrix.tocsr()

def scale(
        x: NDArray, 
        range: Union[List, NDArray], 
        mask: Union[NDArray[np.bool_],None] = None
) -> Tuple[NDArray, float, float]:
    
    if mask is None:
        mask = np.ones_like(x,dtype=bool)
    x_max = x[mask].max()
    x_min = x[mask].min()
    if x_max == x_min:
        warnings.warn("scale: x_max == x_min, set x = x_min", RuntimeWarning)
        a = 0
        b = range[0]
    else:
        if range[1] == range[0]:
            a = 0
            b = range[0]
        else:
            a = (range[1]-range[0])/(x_max-x_min)
            b = range[0]-a*x_min
    return a*x+b,a,b

def create_heatmap_grid(positions, nx=50, ny=50):
    """
    根据粒子坐标范围生成二维网格
    :param positions: 粒子坐标数组，形状为 (N, 2)
    :param nx, ny: 网格在 x 和 y 方向的分辨率
    :return: 网格边界坐标和网格中心坐标
    """
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    
    # 扩展边界避免粒子落在网格外
    x_edges = np.linspace(x_min - 1e-5, x_max + 1e-5, nx + 1)
    y_edges = np.linspace(y_min - 1e-5, y_max + 1e-5, ny + 1)
    
    # 网格中心坐标（用于热力图显示）
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    return x_edges, y_edges, x_centers, y_centers

def compute_grid_pressure(positions, pressure, nx=50, ny=50):
    """
    计算每个网格单元的平均压力
    :return: 平均压力矩阵，形状为 (ny, nx)
    """
    x_edges, y_edges, x_centers, y_centers = create_heatmap_grid(positions, nx, ny)
    
    # 二维直方图统计（权重为粒子压力）
    pressure_grid, _, _ = np.histogram2d(
        positions[:, 1],  # 注意：histogram2d 输入顺序是 (y, x)
        positions[:, 0],
        bins=(y_edges, x_edges),
        weights=pressure
    )
    
    # 计算每个网格的粒子数
    count_grid, _, _ = np.histogram2d(
        positions[:, 1],
        positions[:, 0],
        bins=(y_edges, x_edges)
    )
    
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_pressure = pressure_grid / count_grid
    avg_pressure[np.isnan(avg_pressure)] = 0 #无粒子的网格设为0
    return avg_pressure, x_centers, y_centers