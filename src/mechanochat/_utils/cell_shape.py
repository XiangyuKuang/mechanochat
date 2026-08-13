from typing import Optional, Union, TYPE_CHECKING, List
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components, breadth_first_order
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.axes import Axes
from collections import defaultdict
from tqdm import tqdm

if TYPE_CHECKING:
    from ..SEM import SEM2

# to-do:
# compute norm
class AlphaShape():
    """
    Alpha-shape for element representation of cells.

    Attributes
    ----------
    x : np.ndarray
        Input points with shape (n_points, 2).
    alpha : Optional[float]
        Current alpha radius value.
    alpha_best : Optional[float]
        Optimized alpha radius value.
    simplices : np.ndarray
        Delaunay triangulation simplices.
    neighbors : np.ndarray
        Neighboring simplices indices.
    dmax : np.ndarray
        Maximum edge length of each simplex.
    edge : np.ndarray
        Unique edges in the triangulation.
    edge_counts : np.ndarray
        Occurrence count of each edge.
    tri2edge_I : np.ndarray
        Mapping from simplices to edges.
    alpha_range : list[float]
        Valid alpha range [min_edge, max_edge].
    simplices_adjacency_matrix : csr_matrix
        Adjacency matrix of simplices.
    is_boundary : np.ndarray
        Boolean array marking boundary edges.
    valid_simplices : np.ndarray
        Boolean array marking valid simplices.
    """

    x: np.ndarray
    """Point coordinates"""
    n_points: int
    """Number of points"""
    simplices: np.ndarray
    """Delaunay triangulation simplices, shape=(n_simplices,3), dtype=np.int32"""
    neighbors: np.ndarray
    """Neighboring simplices indices, shape=(n_simplices,3), dtype=np.int32"""
    simplices_adjacency_matrix: csr_matrix
    """Simplices adjacency matrix, shape=(n_simplices,n_simplices)"""
    dmax: np.ndarray
    """Maximum edge length of each simplex, shape=(n_simplices)"""
    edge : np.ndarray
    """Unique edges in the triangulation"""
    is_boundary : np.ndarray
    """Boolean array marking boundary edges"""
    valid_simplices : np.ndarray
    """Boolean array marking shape simplices"""
    alpha: float
    """Current alpha radius"""
    alpha_best: float
    """Optimized alpha radius"""
    
    def __init__(
            self,
            x: np.ndarray, 
            alpha: Optional[float] = None,
            no_hole: bool = True,
            ns: int=0,
            r: float=0.0
    ) -> None:
        """
        Initialize the alpha shape for a given set of points.
        
        Parameters
        ------
        x : np.ndarray
            2D coordinates of the point set.
        alpha : Optional[float]
            Initial alpha radius. If None, the alpha radius will be optimized to ensure the shape enclosing all of the points, having only one region.
        no_hole : bool,
            If True, the alpha radius optimization will include no holes in shape as the objective.
        ns : int default=0
            Number of 
        r : float default=0
        """
        
        if (ns > 0) and (r > 0):
            # add points around elements
            theta = np.linspace(0, 2*np.pi, ns+1)[:-1]
            xs = r*np.column_stack([np.cos(theta),np.sin(theta)])
            x = np.repeat(x, ns, axis=0) + np.tile(xs,(x.shape[0],1))
        self.x = x
        self.n_points = self.x.shape[0]
        tri = Delaunay(x) # perform Delaunay triangulation on the points
        self.simplices = tri.simplices # int32,
        self.neighbors = tri.neighbors # int32,
        self.dmax = np.array([max(pdist(x[simplex])) for simplex in self.simplices])
        # Get unique edges, their indices, and edge counts
        all_edge = np.concatenate([tri2edge(sorted(simplex)) for simplex in self.simplices]) # all edges in the Delaunay triangulation
        self.edge, I, self.edge_counts = np.unique(all_edge,axis=0,return_inverse=True,return_counts=True) # unique edges in the Delaunay triangulation, edges indices, edge counts
        self.tri2edge_I = I.reshape(-1,3) # map triangle to their edges
        
        # Range of alpha radius
        self.alpha_range = [np.min(self.dmax),np.max(self.dmax)]
        
        # Simplices adjacency matrix
        n_simplices = len(self.simplices)
        simplex_indices, neighbor_offsets = np.where(self.neighbors!= -1) # method2 310 µs ± 201 ns per loop
        neighbor_indices = self.neighbors[simplex_indices, neighbor_offsets]
        rows = np.concatenate([simplex_indices, neighbor_indices])
        cols = np.concatenate([neighbor_indices, simplex_indices])
        coo_adj = csr_matrix((np.ones_like(rows), (rows, cols)), shape=(n_simplices, n_simplices)).tocoo()
        coo_adj.sum_duplicates()
        self.simplices_adjacency_matrix = coo_adj.tocsr()
        
        # simplices_adjacency_matrix = np.zeros((n_simplices, n_simplices), dtype=int) # method0 2.65 ms ± 2.62 µs per loop 
        # for simplex_idx, neighbors in enumerate(self.neighbors):
        #     for neighbor_idx in neighbors:
        #         if neighbor_idx != -1:
        #             simplices_adjacency_matrix[simplex_idx, neighbor_idx] = 1
        #             simplices_adjacency_matrix[neighbor_idx, simplex_idx] = 1
        # self.simplices_adjacency_matrix = csr_matrix(simplices_adjacency_matrix) # stored as csr_matrix
        
        # adj = lil_matrix((n_simplices, n_simplices), dtype=int) # method1 16.7 ms ± 31.2 µs per loop
        # for i, neigh in enumerate(self.neighbors):
        #     valid_neigh = neigh[neigh != -1]
        #     adj[i, valid_neigh] = 1
        # self.simplices_adjacency_matrix = adj.tocsr() # stored as csr_matrix
        
        # If no alpha radius is provided, use optimized alpha radius
        self.alpha = alpha
        self.alpha_best = None
        if self.alpha is None:
            self.optimize_alpha1(no_hole)
            self.alpha = self.alpha_best
        self.update(self.alpha) # update the alpha shape
        
    def __repr__(self) -> str:
        return f'alpha-shape with alpha = {self.alpha}, {len(self.simplices)} simplices, {self.n_points} points'
    
    def update(self, alpha: float) -> None:
        """
        Update the alpha shape based on the given alpha radius. Calculate valid simplices and boundary edges.
        
        Parameters
        ------
        alpha : float
            Alpha radius used to determine valid simplices and boundaries.
        """
        self.alpha = alpha
        # Determine boundary edges, whose counts = 1
        edge_counts_update = self.edge_counts.copy()
        for i in self.tri2edge_I[self.dmax>alpha].flatten():
            edge_counts_update[i] -= 1
        self.is_boundary = edge_counts_update == 1
        # Mark simplices (maximum edge length< alpha radius) as valid
        self.valid_simplices = self.dmax<=alpha
        
    def get_shape_info(self):
        simplices_adjacency_matrix = self.simplices_adjacency_matrix.toarray()
        cover_points = np.unique(self.simplices[self.valid_simplices].flatten())
        n_components, labels = connected_components(csr_matrix(simplices_adjacency_matrix[self.valid_simplices,:][:,self.valid_simplices]))
        return cover_points, n_components, labels

    def optimize_alpha(self, no_hole: bool = True, return_alpha: bool = False) -> Union[float, None]:
        """
        Optimize the alpha radius using binary search to ensure that the shape contains all points.
        
        Optionally, ensures that the shape has no holes.

        Parameters
        -------
        no_hole : bool, default=True
            If True, the optimization ensures that no holes exist in the shape.
        return_alpha : bool, default=False
            If True, return the optimized alpha value.
        
        Return
        -------
        alpha_best : Union[float, None]
            Optimized alpha value if `return_alpha=True`, otherwise None.
        """
        # All points in one connected region
        def allpoints_1region(simplices,outer,adjacency_matrix,mask,n_points):
            I = np.where(mask)[0]
            n_components, _ = connected_components(adjacency_matrix[I,:][:,I], directed=False)
            return n_components == 1 and np.unique(simplices[mask].flatten()).shape[0] == n_points
        
        # All points in one connected region and no hole
        def allpoints_1region_0hole(simplices,outer,adjacency_matrix,mask,n_points):
            I = np.where(mask)[0]
            I1 = np.where(~mask)[0]
            n_components, _ = connected_components(adjacency_matrix[I,:][:,I], directed=False)
            _, labels = connected_components(adjacency_matrix[I1,:][:,I1], directed=False)
            return n_components == 1 and np.unique(simplices[mask].flatten()).shape[0] == n_points and np.setdiff1d(np.unique(labels),np.unique(labels[outer[~mask]])).shape[0]==0
        
        # Select the optimization objective
        opt_obj = allpoints_1region_0hole if no_hole else allpoints_1region
        
        #  Initialize the best alpha as the maximum possible alpha
        alpha_low, alpha_high = self.alpha_range[0], self.alpha_range[1]
        alpha_best = alpha_high

        tol=1e-3 # Tolerance for stopping the search
        max_iterations=10 # Maximum number of iterations for binary search
        # n_points = self.x.shape[0]
        outer = np.any(self.neighbors==-1,axis=1)# identify the outer simplices (those touching the boundary)
        # Binary search to find the optimal alpha value
        for _ in range(max_iterations):
            alpha_mid = (alpha_low + alpha_high) / 2# Midpoint alpha value
            mask = self.dmax <= alpha_mid # Mark simplices (maximum edge length< alpha_mid) as valid
            if opt_obj(self.simplices,outer,self.simplices_adjacency_matrix,mask,self.n_points):
                # Narrow the search to lower values
                alpha_best = alpha_mid
                alpha_high = alpha_mid
            else:
                # Narrow the search to higher values
                alpha_low = alpha_mid
            if alpha_high - alpha_low < tol: # stop searching
                break
        self.alpha_best = alpha_best
        if return_alpha:
            return alpha_best
    
    def optimize_alpha1(self, no_hole: bool = True, return_alpha: bool = False) -> Union[float, None]:
        # problem : alpha value - n boundary is not Monotonic: increase alpha may create holes
        """
        Optimize the alpha radius using binary search based on boundary edges' connectivity.
        
        The alpha shape is considered valid if:
        - All points are covered by the valid simplices.
        - There is exactly one closed boundary loop (no holes).
        """
        alpha_low, alpha_high = self.alpha_range[0], self.alpha_range[1]
        alpha_best = alpha_high
        tol = 1e-3
        max_iterations = 10

        for _ in range(max_iterations):
            alpha_mid = (alpha_low + alpha_high) / 2
            self.update(alpha_mid)  # Update valid_simplices and is_boundary

            # Check if all points are covered
            cover_points = np.unique(self.simplices[self.valid_simplices].flatten())
            all_covered = cover_points.shape[0] == self.n_points

            # Get boundary edges and count closed loops
            boundaries = self.get_boundary_vertices1()

            # Determine if conditions are met
            if no_hole:
                # Require exactly one boundary loop and all points covered
                condition_met = all_covered and (len(boundaries) == 1)
            else:
                # Original logic (fallback, adjust if necessary)
                # Here, we just ensure all points are covered (no hole check)
                condition_met = all_covered

            if condition_met:
                alpha_best = alpha_mid
                alpha_high = alpha_mid
            else:
                alpha_low = alpha_mid

            if alpha_high - alpha_low < tol:
                break

        self.alpha_best = alpha_best
        if return_alpha:
            return alpha_best

    def close_hole(self):
        """
        Close holes in the alpha shape by marking invalid simplices corresponding to holes as valid.
        """
        # Identify outer simplices (those touching the boundary)
        outer = np.any(self.neighbors == -1, axis=1)
        # Compute valid simplices for current alpha shape
        mask = self.valid_simplices
        # Identify connected components among invalid simplices (corresponding to holes)
        _, labels = connected_components(csr_matrix(self.simplices_adjacency_matrix[~mask ,:][:,~mask ]))
        invalid_simplices_idx = np.where(~mask)[0]
        # Find the connected components corresponding to holes
        hole_labels = np.setdiff1d(np.unique(labels), np.unique(labels[outer[~mask]]))
        if hole_labels.size > 0:
            # Loop through the hole_labels and close the holes by marking those simplices as valid
            for hole_label in hole_labels:
                hole_simplices = np.where(labels == hole_label)[0]
                # Mark the simplices corresponding to the hole as valid
                self.valid_simplices[invalid_simplices_idx[hole_simplices]] = True
        # update boundary after closing holes
        edge_counts_update = self.edge_counts.copy()
        for i in self.tri2edge_I[~self.valid_simplices].flatten():
            edge_counts_update[i] -= 1
        self.is_boundary = edge_counts_update == 1

    def get_boundary_vertices(self) -> list:
        boundary_edges = self.edge[self.is_boundary]
        boundary_points = [boundary_edges[0][0], boundary_edges[0][1]]
        boundary_edges = np.delete(boundary_edges, 0, axis=0)
        while len(boundary_edges) > 1:
            last_point = boundary_points[-1]
            match_idx = np.where(boundary_edges == last_point )[0]
            if len(match_idx) == 0:
                # if len(boundary_edges)>1:
                #     print('boundary not finished')
                break  # 如果没有找到下一个相接的点，说明没有闭合
            match_edge = boundary_edges[match_idx[0]]
            boundary_edges = np.delete(boundary_edges, match_idx[0], axis=0)
            # 将匹配到的边的另一个点加入序列
            if match_edge[0] == last_point:
                boundary_points.append(match_edge[1])
            else:
                boundary_points.append(match_edge[0])
        return boundary_points
    
    def get_boundary_vertices1(self):
        boundary_edges = self.edge[self.is_boundary]
        adj = defaultdict(set)
        for u, v in boundary_edges:
            adj[u].add(v)
            adj[v].add(u)

        boundaries = []

        while adj:
            # 获取起始点（任取一个未处理的点）
            start = next(iter(adj))
            current = start
            
            # 初始化路径并处理第一条边
            next_node = adj[current].pop()
            if not adj[current]:
                del adj[current]
            path = [current, next_node]
            
            # 移除反向边
            adj[next_node].remove(current)
            if not adj[next_node]:
                del adj[next_node]
            current = next_node
            
            # 循环构建闭合路径
            while True:
                prev = path[-2] if len(path) >= 2 else None
                # 获取当前节点的候选邻接点（排除前驱节点）
                candidates = adj[current] - {prev} if prev is not None else adj[current]
                
                if not candidates:
                    break  # 理论上不会触发，除非输入不合法
                
                next_node = candidates.pop()
                path.append(next_node)
                
                # 移除当前边及其反向边
                adj[current].remove(next_node)
                if not adj[current]:
                    del adj[current]
                adj[next_node].remove(current)
                if not adj[next_node]:
                    del adj[next_node]
                
                current = next_node
                
                # 检查是否闭合
                if current == start:
                    break
            
            # 确保路径闭合（应对非正常情况）
            if path[0] != path[-1]:
                path.append(path[0])
            
            boundaries.append(path)
        return boundaries
    
    def get_boundary(self, close = True) -> NDArray:
        boundary_points = self.get_boundary_vertices()
        if close:
            boundary_points = np.append(boundary_points,boundary_points[0])
        return self.x[boundary_points]
    
    def get_boundarys(self, close = True) -> List[NDArray]:
        boundaries = self.get_boundary_vertices1()
        xb = []
        for ptr in boundaries:
            xb.append(self.x[ptr])
        return xb
    
    def get_simplices(self) -> NDArray:
        return self.simplices[self.valid_simplices]
    
    def get_area(self) -> np.floating:
        area = 0.
        for spx in self.get_simplices():
            area+=np.cross(self.x[spx[1]]-self.x[spx[0]],self.x[spx[2]]-self.x[spx[0]])
        return area/2

    def plot_shape(self, alpha: Optional[float] = None, fill:bool = True, ax: Optional[Axes] = None, **kwargs):
        """
        Plot the alpha shape. Optionally update the alpha value before plotting.
        
        Parameters
        ----------
        alpha : float, optional
            The alpha value to update the shape with before plotting.
        fill : bool
            If True, the shape will be filled with color. Default is True.
        ax : sAxes, optional
            The axis to plot on. If None, a new axis will be created.
        **kwargs : dict, optional
            Additional styling arguments for the boundary and face (boundarywidth, boundarycolor, facecolor, facealpha).
        """
        if alpha is not None:
            print(f'update alpha to {alpha}')
            self.update(alpha)
        if ax is None:
            fig,ax=plt.subplots()
        line_kwargs = dict()
        poly_kwargs = dict(facecolors='gray',edgecolors='gray')
        for k, v in kwargs.items():
            if k == 'boundarywidth':
                line_kwargs['linewidths'] = v
            elif k == 'boundarycolor':
                line_kwargs['colors'] = v
            elif k == 'facecolor':
                poly_kwargs['facecolors'] = v
                poly_kwargs['edgecolors'] = v
            elif k == 'facealpha':
                poly_kwargs['alpha'] = v
        boundary_x = [self.get_boundary()]
        lc = LineCollection(boundary_x,**line_kwargs)
        ax.add_collection(lc)
        if fill:
            poly = PolyCollection(boundary_x, **poly_kwargs)
            ax.add_collection(poly)
        ax.autoscale()
        return ax
    
def tri2edge(simplex):
    """map a triangle into its edges"""
    return [[simplex[0],simplex[1]],[simplex[0],simplex[2]],[simplex[1],simplex[2]]]

class cell_tri():
    contact_matrix : csr_matrix
    
    def __init__(self, sem: "SEM2"):
        tri = Delaunay(sem.xe)
        self.nc = sem.nc
        self.ne = sem.ne
        self.xe = sem.xe
        self.ecid = sem.ecid
        self.sem = sem
        dmax = np.array([max(pdist(self.xe[simplex])) for simplex in tri.simplices])
        dth = sem._get_e_radius()*1.5
        mask = dmax<=dth
        keep_idx = np.where(mask)[0] 
        old2new = -np.ones(tri.simplices.shape[0], dtype=int)
        old2new[keep_idx] = np.arange(keep_idx.size)
        raw_neighbors = tri.neighbors[mask]
        neighbors = old2new[raw_neighbors]
        neighbors[raw_neighbors==-1] = -1

        self.simplices = tri.simplices[mask]
        self.neighbors = neighbors
        self.scid = sem.ecid[self.simplices] # simplex的element所属的cid
        self.scid_u = [np.unique(scid_i) for scid_i in self.scid]
        # tri dict
        self.cell_tri_dict = defaultdict(list)
        self.contact_tri_dict = defaultdict(list)
        for i, u in enumerate(self.scid_u):
            if len(u) == 1:  # 内部三角形
                self.cell_tri_dict[u[0]].append(i)
            else:
                self.contact_tri_dict[tuple(np.sort(u))].append(i)
        
        # Get unique edges, their indices, and edge counts
        all_edge = np.concatenate([tri2edge(sorted(simplex)) for simplex in self.simplices]) # all edges in the Delaunay triangulation
        self.edge, I = np.unique(all_edge,axis=0,return_inverse=True) # unique edges in the Delaunay triangulation, edges indices, edge counts
        self.tri2edge_I = I.reshape(-1,3) # map triangle to their edges

        e_contact_matrix = lil_matrix((self.ne, self.ne))
        for simplex in self.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    e_contact_matrix[simplex[i], simplex[j]] = 1
                    e_contact_matrix[simplex[j], simplex[i]] = 1
        self.e_contact_matrix = e_contact_matrix.tocsr()
        self.cell_boundary = []

    def compute_shape(self):
        """计算每个细胞的边界"""
        dmax = np.array([max(pdist(self.xe[simplex])) for simplex in self.simplices])
        dth = np.percentile(dmax,95)
        cell_tri_list = self.get_cell_tri_i()
        cell_boundary = []
        for tri_i in cell_tri_list:
            if len(tri_i)>0:
                tri_i = np.array(tri_i)
                I = dmax[tri_i] < dth
                ctri_i_edge_I = self.tri2edge_I[tri_i[I]].flatten()
                edge_I, edge_counts = np.unique(ctri_i_edge_I, axis=0, return_counts=True)
                is_boundary = edge_counts == 1
                boundary_edges = self.edge[edge_I[is_boundary]]
                adj = defaultdict(set)
                for u, v in boundary_edges:
                    adj[u].add(v)
                    adj[v].add(u)

                boundaries = []
                
                while adj:
                    # 获取起始点（任取一个未处理的点）
                    start = next(iter(adj))
                    current = start
                    
                    # 初始化路径并处理第一条边
                    next_node = adj[current].pop()
                    if not adj[current]:
                        del adj[current]
                    path = [current, next_node]
                    
                    # 移除反向边
                    adj[next_node].remove(current)
                    if not adj[next_node]:
                        del adj[next_node]
                    current = next_node
                    
                    # 循环构建闭合路径
                    while True:
                        prev = path[-2] if len(path) >= 2 else None
                        # 获取当前节点的候选邻接点（排除前驱节点）
                        candidates = adj[current] - {prev} if prev is not None else adj[current]
                        
                        if not candidates:
                            break  # 理论上不会触发，除非输入不合法
                        
                        next_node = candidates.pop()
                        path.append(next_node)
                        
                        # 移除当前边及其反向边
                        adj[current].remove(next_node)
                        if not adj[current]:
                            del adj[current]
                        adj[next_node].remove(current)
                        if not adj[next_node]:
                            del adj[next_node]
                        
                        current = next_node
                        
                        # 检查是否闭合
                        if current == start:
                            break
                    
                    # 确保路径闭合（应对非正常情况）
                    if path[0] != path[-1]:
                        path.append(path[0])
                    
                    boundaries.append(path)
                if len(boundaries)>1:
                    cell_boundary.append(boundaries[np.argmax([len(b) for b in boundaries])])
                elif len(boundaries)==1:
                    cell_boundary.append(boundaries[0])
                else:
                    cell_boundary.append([])
            else:
                cell_boundary.append([])
            self.cell_boundary = cell_boundary

    def compute_contact(self):
        row_indices = []
        col_indices = []
        data = []
        for i in range(self.nc):
            for j in range(i+1,self.nc):
                tri_ij_I = self.contact_tri_dict[tuple([i,j])]
                if len(tri_ij_I)>0:
                    tri_ij = self.simplices[tri_ij_I]
                    contact_eid = np.unique(np.array(tri_ij))
                    contact_ceid = self.ecid[contact_eid]
                    a_i = (contact_ceid == i).sum()
                    a_j = len(contact_ceid) - a_i
                    a_ij = (a_i+a_j)/2
                    row_indices.append(i)
                    col_indices.append(j)
                    data.append(a_ij)
        contact_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(self.nc, self.nc))
        self.contact_matrix = contact_matrix+contact_matrix.T
        
    def get_cell_tri_i(self,cids = None):
        i = []
        if cids is None:
            cids = range(self.nc)
        elif np.isscalar(cids):
            cids = [cids]
        for cid in cids:
            i.append(self.cell_tri_dict[cid])
        return i
    
    def get_contact_tri_i(self,cid_ps):
        i = []
        if np.isscalar(cid_ps[0]):
            i.append(self.contact_tri_dict[tuple(sorted(cid_ps))])
        else:
            for cid_p in cid_ps:
                i.append(self.contact_tri_dict[tuple(sorted(cid_p))])
        return i

    def plot_cell_tri(self,cids=None,ax = None):
        if ax is None:
            fig,ax = plt.subplots()
        I = self.get_cell_tri_i(cids)
        tri_list = []
        for I_i in I:
            for tri_i in self.simplices[I_i]:
                tri_list.append(self.xe[tri_i])
        polyc = PolyCollection(tri_list)
        ax.add_collection(polyc)
        ax.autoscale(tight=True)
        ax.set_aspect('equal', adjustable='box')
        return ax
    
    def plot_cell_boundary(self,cids,ax= None):
        if ax is None:
            fig,ax = plt.subplots()
        boundarys = []
        for i in cids:
            boundarys.append(self.xe[self.cell_boundary[i]])
        polyc = PolyCollection(boundarys)
        ax.add_collection(polyc)
        ax.autoscale(tight=True)
        ax.set_aspect('equal', adjustable='box')
        return ax

    def compute_contact_force(self):
        """run sem.compute_force() first"""
        fx_csr, fy_csr = self.sem.f_pairs_tocsr()
        ceidn = self.sem.ceidn
        f_e_inter = np.zeros((self.ne,2))
        I = np.zeros(self.ne).astype(bool)
        for i in range(self.ne):
            esid_i = np.where(self.simplices == i)[0]
            esid_u_i = np.unique(self.scid[esid_i])
            I[i] = len(esid_u_i)>1
            if I[i]:
                for cid_j in esid_u_i[esid_u_i != self.ecid[i]]:
                    f_e_inter[i,0] += fx_csr[i,ceidn[cid_j]:ceidn[cid_j+1]].sum()
                    f_e_inter[i,1] += fy_csr[i,ceidn[cid_j]:ceidn[cid_j+1]].sum()
        f_e_inter_a = np.linalg.norm(f_e_inter,axis=1)

        deltax = self.sem.deltax
        scale = self.sem.scale
        self.contact_midedge_dict = defaultdict(list)
        self.contact_f_dict = defaultdict(list)
        self.junction_midedge_dict = defaultdict(list)
        self.junction_f_dict = defaultdict(list)
        for ij in self.contact_tri_dict:
            tri_ij = self.simplices[self.contact_tri_dict[ij]]
            if len(ij)==2:
                for tri_ij0 in tri_ij:
                    ecid0 = self.ecid[tri_ij0]
                    f_e_inter_a0 = f_e_inter_a[tri_ij0]
                    mid = []
                    ten_mid = []
                    for vi,vj in [(0,1), (1,2), (2,0)]:
                        ci = ecid0[vi]
                        cj = ecid0[vj]
                        if ci!=cj:
                            ei = tri_ij0[vi]
                            ej = tri_ij0[vj]
                            fa_i = f_e_inter_a0[vi]
                            fa_j = f_e_inter_a0[vj]
                            mid.append((self.xe[ei] + self.xe[ej]) / 2*scale+deltax)
                            ten_mid.append((fa_i+fa_j)/2)
                    self.contact_f_dict[ij].append(ten_mid)
                    self.contact_midedge_dict[ij].append(mid)
            elif len(ij)==3:
                for tri_ij0 in tri_ij:
                    center = self.xe[tri_ij0].mean(axis=0)
                    f_e_inter_a0 = f_e_inter_a[tri_ij0]
                    seg = []
                    junction_ten = []
                    for vi,vj in [(0,1), (1,2), (2,0)]:
                        fa_i = f_e_inter_a0[vi]
                        fa_j = f_e_inter_a0[vj]
                        seg.append([center*scale+deltax,(self.xe[tri_ij0[vi]] + self.xe[tri_ij0[vj]]) / 2*scale+deltax])
                        junction_ten.append((fa_i+fa_j)/2)
                    self.junction_midedge_dict[ij].extend(seg)
                    self.junction_f_dict[ij].extend(junction_ten)
        print('add .contact_midedge_dict')
        print('add .contact_f_dict')
        print('add .junction_midedge_dict')
        print('add .junction_f_dict')

    def contact_tension(self, scaling = True):
        """require sem.e_sigma, run sem.compute_tensor() first"""
        deltax = self.sem.deltax
        scale = self.sem.scale
        self.contact_midedge_dict = defaultdict(list)
        self.contact_ten_dict = defaultdict(list)
        for ij in self.contact_tri_dict:
            tri_ij = self.simplices[self.contact_tri_dict[ij]]
            if len(ij)==2:
                for tri_ij0 in tri_ij:
                    ecid0 = self.ecid[tri_ij0]
                    a=np.where(ecid0 == ij[0])[0]
                    b=np.where(ecid0 == ij[1])[0]
                    if len(a)>len(b):
                        vi,vj = a
                        vk = b[0]
                    else:
                        vi,vj = b
                        vk = a[0]
                    ei = tri_ij0[vi]
                    ej = tri_ij0[vj]
                    ek = tri_ij0[vk]
                    vt = self.xe[ej]-self.xe[ei]
                    mid1 = (self.xe[ei] + self.xe[ek]) / 2
                    mid2 = (self.xe[ej] + self.xe[ek]) / 2
                    if scaling:
                        mid1 = mid1*scale+deltax
                        mid2 = mid2*scale+deltax
                    self.contact_midedge_dict[ij].append([mid1, mid2])
                    vt /= np.linalg.norm(vt)
                    c = vt[1]**2-vt[0]**2
                    d = 4*vt[0]*vt[1]
                    ten_i = 0
                    for j in [ei,ej,ek]:
                        sigma_j = self.sem.e_sigma[j]
                        ten_i += c*(sigma_j[0,0]-sigma_j[1,1])-d*sigma_j[0,1]
                    self.contact_ten_dict[ij].append(ten_i/3)
        print("add .contact_midedge_dict")
        print("add .contact_ten_dict")
        
    # def pressure(self):
    
    def cell_tension(self):
        """require sem.e_sigma, run sem.compute_tensor() first"""
        cid_list = [] # id of cells that are detected boundaries, corresponding to ten_list
        ten_list = [] # tension of each cell
        for i in range(self.nc):
            eid_ib = self.cell_boundary[i].copy()
            if len(eid_ib)>0:
                if eid_ib[-1]!=eid_ib[0]:
                    eid_ib.append(eid_ib[0])
                # xe_ib = self.xe[eid_ib]
                ten_i =  np.zeros(len(eid_ib)-1)
                eid_ib1 = np.roll(eid_ib[:-1],1)
                eid_ib2 = np.roll(eid_ib[:-1],-1)
                vt = self.xe[eid_ib1]-self.xe[eid_ib2]
                vt /= np.linalg.norm(vt,axis=1, keepdims=True)
                for n,j in enumerate(eid_ib[:-1]):# todo vectorize tension and sigma computation
                    sigma_j = self.sem.e_sigma[j]
                    ten_i[n] = (vt[n,1]**2-vt[n,0]**2)*(sigma_j[0,0]-sigma_j[1,1])-4*vt[n,0]*vt[n,1]*sigma_j[0,1]
                # xe_b_rs = xe_ib.reshape(-1, 1, 2)
                ten_list.append(ten_i)
                cid_list.append(i)
        self.cell_ten_dict = dict(zip(cid_list,ten_list))#, xe_b, searching
        print("add .cell_ten_dict")

    def cell2contact_tension(self,scaling = True):
        """run cell_tension() first"""
        deltax = self.sem.deltax
        scale = self.sem.scale
        self.contact_midedge_dict1 = defaultdict(list)
        self.contact_ten_dict1 = defaultdict(list)
        for ij in self.contact_tri_dict:
            tri_ij = self.simplices[self.contact_tri_dict[ij]]
            if len(ij)==2:
                for tri_ij0 in tri_ij:
                    ecid0 = self.ecid[tri_ij0]
                    mid = []
                    ten_mid = []
                    for vi,vj in [(0,1), (1,2), (2,0)]:
                        ci = ecid0[vi]
                        cj = ecid0[vj]
                        if ci!=cj:
                            ei = tri_ij0[vi]
                            ej = tri_ij0[vj]
                            ten_ei_I = np.where(np.array(self.cell_boundary[ci]) == ei)[0]
                            ten_ej_I = np.where(np.array(self.cell_boundary[cj]) == ej)[0]
                            mid_ij = (self.xe[ei] + self.xe[ej]) / 2
                            if scaling:
                                mid_ij = mid_ij*scale+deltax
                            mid.append(mid_ij)
                            if len(ten_ei_I)>0 and len(ten_ej_I)>0:
                                ten_mid.append(self.cell_ten_dict[ci][ten_ei_I[0]] + self.cell_ten_dict[cj][ten_ej_I[0]])
                            else:
                                ten_mid.append(np.nan)
                    self.contact_ten_dict1[ij].append(ten_mid)
                    self.contact_midedge_dict1[ij].append(mid)
        print("add .contact_midedge_dict1")
        print("add .contact_ten_dict1")

class cell_tri_dev():

    def __init__(self, sem: "SEM2", cid_list: Optional[NDArray[np.int_]]):
        self.sem = sem
        
        self.subset_mode = cid_list is not None
        if self.subset_mode:
            cid_list = np.asarray(cid_list)
            self._cid_original = cid_list
            # Element mask: 只选择属于指定细胞的elements
            e_mask = np.isin(sem.ecid, cid_list)
            e_indices_original = np.where(e_mask)[0]  # 原始element索引
            
            # 提取子集坐标
            xe_subset = sem.xe[e_mask]
            ecid_subset = sem.ecid[e_mask]  # 保持原始cell ID
            
            # Element索引映射: 原始 -> 新
            self._old2new_e = -np.ones(sem.ne, dtype=int)
            self._old2new_e[e_mask] = np.arange(e_mask.sum())
            # Element索引映射: 新 -> 原始
            self._new2old_e = e_indices_original
            
            self.nc = len(cid_list)
            self.ne = xe_subset.shape[0]
            self.xe = xe_subset
            self.ecid = ecid_subset  # 保持原始cell ID
            self._e_mask = e_mask
        else:
            self._cid_original = np.arange(sem.nc)
            self.nc = sem.nc
            self.ne = sem.ne
            self.xe = sem.xe
            self.ecid = sem.ecid
            self._old2new_e = np.arange(sem.ne)
            self._new2old_e = np.arange(sem.ne)
            self._e_mask = np.ones(sem.ne, dtype=bool)

        tri = Delaunay(self.xe)
        dmax = np.array([max(pdist(self.xe[simplex])) for simplex in tri.simplices])
        dth = sem._get_e_radius()*1.5
        mask = dmax<=dth
        keep_idx = np.where(mask)[0] 
        old2new = -np.ones(tri.simplices.shape[0], dtype=int)
        old2new[keep_idx] = np.arange(keep_idx.size)
        raw_neighbors = tri.neighbors[mask]
        neighbors = old2new[raw_neighbors]
        neighbors[raw_neighbors==-1] = -1

        self.simplices = tri.simplices[mask]
        self.neighbors = neighbors
        self.scid = self.ecid[self.simplices] # simplex的element所属的cid
        self.scid_u = [np.unique(scid_i) for scid_i in self.scid]

        # tri dict - 使用原始cell ID作为key
        self.cell_tri_dict = defaultdict(list)
        self.contact_tri_dict = defaultdict(list)
        for i, u in enumerate(self.scid_u):
            if len(u) == 1:  # 内部三角形
                self.cell_tri_dict[u[0]].append(i)
            else:
                self.contact_tri_dict[tuple(np.sort(u))].append(i)
        
        # Get unique edges, their indices, and edge counts
        all_edge = np.concatenate([tri2edge(sorted(simplex)) for simplex in self.simplices]) # all edges in the Delaunay triangulation
        self.edge, I = np.unique(all_edge,axis=0,return_inverse=True) # unique edges in the Delaunay triangulation, edges indices, edge counts
        self.tri2edge_I = I.reshape(-1,3) # map triangle to their edges

        e_contact_matrix = lil_matrix((self.ne, self.ne))
        for simplex in self.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    e_contact_matrix[simplex[i], simplex[j]] = 1
                    e_contact_matrix[simplex[j], simplex[i]] = 1
        self.e_contact_matrix = e_contact_matrix.tocsr()
        self.cell_boundary = []

    def _get_original_cids(self) -> NDArray:
        """返回分析的原始cell IDs"""
        return self._cid_original
    
    def get_cell_tri_i(self, cids = None):
        """获取指定细胞的内部三角形索引，cids为原始cell ID"""
        i = []
        if cids is None:
            cids = self._cid_original
            # cids = range(self.nc)
        elif np.isscalar(cids):
            cids = [cids]
        for cid in cids:
            # if cid in self.cell_tri_dict:
            #     i.append(self.cell_tri_dict[cid])
            i.append(self.cell_tri_dict[cid])
        return i
    
    def get_contact_tri_i(self,cid_ps):
        """获取指定细胞对的接触三角形索引，cid_ps使用原始cell ID"""
        i = []
        if np.isscalar(cid_ps[0]):
            i.append(self.contact_tri_dict[tuple(sorted(cid_ps))])
        else:
            for cid_p in cid_ps:
                i.append(self.contact_tri_dict[tuple(sorted(cid_p))])
        return i
    
    def compute_shape(self):
        """计算每个细胞的边界"""
        dmax = np.array([max(pdist(self.xe[simplex])) for simplex in self.simplices])
        dth = np.percentile(dmax, 95)
        
        cell_boundary = {}  # 使用dict，key为原始cell ID
        
        for cid in self._cid_original:
            tri_i = self.cell_tri_dict.get(cid, [])
            if len(tri_i) > 0:
                tri_i = np.array(tri_i)
                I = dmax[tri_i] < dth
                if I.sum() == 0:
                    cell_boundary[cid] = []
                    continue
                    
                ctri_i_edge_I = self.tri2edge_I[tri_i[I]].flatten()
                edge_I, edge_counts = np.unique(ctri_i_edge_I, axis=0, return_counts=True)
                is_boundary = edge_counts == 1
                boundary_edges = self.edge[edge_I[is_boundary]]
                
                if len(boundary_edges) == 0:
                    cell_boundary[cid] = []
                    continue
                
                adj = defaultdict(set)
                for u, v in boundary_edges:
                    adj[u].add(v)
                    adj[v].add(u)

                boundaries = []
                while adj:
                    # 获取起始点（任取一个未处理的点）
                    start = next(iter(adj))
                    current = start
                    
                    # 初始化路径并处理第一条边
                    next_node = adj[current].pop()
                    if not adj[current]:
                        del adj[current]
                    path = [current, next_node]

                    # 移除反向边
                    adj[next_node].remove(current)
                    if not adj[next_node]:
                        del adj[next_node]
                    current = next_node
                    
                    # 循环构建闭合路径
                    while True:
                        prev = path[-2] if len(path) >= 2 else None
                        # 获取当前节点的候选邻接点（排除前驱节点）
                        candidates = adj.get(current, set()) - {prev} if prev is not None else adj.get(current, set())
                        if not candidates:
                            break  # 理论上不会触发，除非输入不合法
                        
                        next_node = candidates.pop()
                        path.append(next_node)

                        # 移除当前边及其反向边
                        adj[current].remove(next_node)
                        if not adj[current]:
                            del adj[current]
                        if next_node in adj:
                            adj[next_node].remove(current)
                            if not adj[next_node]:
                                del adj[next_node]

                        current = next_node

                        # 检查是否闭合
                        if current == start:
                            break
                    
                    # 确保路径闭合（应对非正常情况）
                    if path[0] != path[-1]:
                        path.append(path[0])
                    
                    boundaries.append(path)
                
                if len(boundaries) > 1:
                    cell_boundary[cid] = boundaries[np.argmax([len(b) for b in boundaries])]
                elif len(boundaries) == 1:
                    cell_boundary[cid] = boundaries[0]
                else:
                    cell_boundary[cid] = []
            else:
                cell_boundary[cid] = []
        
        self.cell_boundary = cell_boundary

    def plot_cell_tri(self,cids=None,ax = None):
        if ax is None:
            fig,ax = plt.subplots()
        I = self.get_cell_tri_i(cids)
        tri_list = []
        for I_i in I:
            for tri_i in self.simplices[I_i]:
                tri_list.append(self.xe[tri_i])
        polyc = PolyCollection(tri_list)
        ax.add_collection(polyc)
        ax.autoscale(tight=True)
        ax.set_aspect('equal', adjustable='box')
        return ax
    
    def plot_cell_boundary(self,cids,ax= None):
        if ax is None:
            fig,ax = plt.subplots()
        boundarys = []
        for i in cids:
            boundarys.append(self.xe[self.cell_boundary[i]])
        polyc = PolyCollection(boundarys)
        ax.add_collection(polyc)
        ax.autoscale(tight=True)
        ax.set_aspect('equal', adjustable='box')
        return ax

    def compute_contact_force(self):
        """计算接触力，需要先运行sem.compute_force()"""
        if self.subset_mode:
            fx_csr, fy_csr = self.sem.f_pairs_tocsr(e_mask=self._e_mask)
        else:
            fx_csr, fy_csr = self.sem.f_pairs_tocsr()
        
        # 重建subset中的ceidn映射
        # ceidn_sub = {}
        # for cid in self._cid_original:
        #     e_in_cell = np.where(self.ecid == cid)[0]
        #     ceidn_sub[cid] = (e_in_cell.min(), e_in_cell.max() + 1) if len(e_in_cell) > 0 else (0, 0)
        
        f_e_inter = np.zeros((self.ne, 2))
        I = np.zeros(self.ne).astype(bool)
        
        for i in range(self.ne):
            esid_i = np.where(self.simplices == i)[0]
            esid_u_i = np.unique(self.scid[esid_i])
            I[i] = len(esid_u_i) > 1
            if I[i]:
                my_cid = self.ecid[i]
                for cid_j in esid_u_i[esid_u_i != my_cid]:
                    # 找到属于cid_j的elements在subset中的索引
                    e_j_indices = np.where(self.ecid == cid_j)[0]
                    f_e_inter[i, 0] += fx_csr[i, e_j_indices].sum()
                    f_e_inter[i, 1] += fy_csr[i, e_j_indices].sum()
        
        f_e_inter_a = np.linalg.norm(f_e_inter, axis=1)
        
        deltax = self.sem.deltax
        scale = self.sem.scale
        
        # 使用原始cell ID作为key
        self.contact_midedge_dict = defaultdict(list)
        self.contact_f_dict = defaultdict(list)
        self.junction_midedge_dict = defaultdict(list)
        self.junction_f_dict = defaultdict(list)

        for ij in self.contact_tri_dict:
            tri_ij = self.simplices[self.contact_tri_dict[ij]]
            if len(ij)==2:
                for tri_ij0 in tri_ij:
                    ecid0 = self.ecid[tri_ij0]
                    f_e_inter_a0 = f_e_inter_a[tri_ij0]
                    mid = []
                    ten_mid = []
                    for vi,vj in [(0,1), (1,2), (2,0)]:
                        ci = ecid0[vi]
                        cj = ecid0[vj]
                        if ci!=cj:
                            ei = tri_ij0[vi]
                            ej = tri_ij0[vj]
                            fa_i = f_e_inter_a0[vi]
                            fa_j = f_e_inter_a0[vj]
                            mid.append((self.xe[ei] + self.xe[ej]) / 2*scale+deltax)
                            ten_mid.append((fa_i+fa_j)/2)
                    self.contact_f_dict[ij].append(ten_mid)
                    self.contact_midedge_dict[ij].append(mid)
            elif len(ij)==3:
                for tri_ij0 in tri_ij:
                    center = self.xe[tri_ij0].mean(axis=0)
                    f_e_inter_a0 = f_e_inter_a[tri_ij0]
                    seg = []
                    junction_ten = []
                    for vi,vj in [(0,1), (1,2), (2,0)]:
                        fa_i = f_e_inter_a0[vi]
                        fa_j = f_e_inter_a0[vj]
                        seg.append([center*scale+deltax,(self.xe[tri_ij0[vi]] + self.xe[tri_ij0[vj]]) / 2*scale+deltax])
                        junction_ten.append((fa_i+fa_j)/2)
                    self.junction_midedge_dict[ij].extend(seg)
                    self.junction_f_dict[ij].extend(junction_ten)
        
        print("add .contact_midedge_dict")
        print("add .contact_f_dict")
        print("add .junction_midedge_dict")
        print("add .junction_f_dict")

    def contact_energy(self, h_int, scaling = True):
        if self.subset_mode:
            h_int = h_int[self._e_mask]
        deltax = self.sem.deltax
        scale = self.sem.scale
        self.contact_midedge_dict = defaultdict(list)
        self.contact_ten_dict = defaultdict(list)

        for ij in self.contact_tri_dict:
            tri_ij = self.simplices[self.contact_tri_dict[ij]]
            if len(ij)==2:
                for tri_ij0 in tri_ij:
                    ecid0 = self.ecid[tri_ij0]
                    a=np.where(ecid0 == ij[0])[0]
                    b=np.where(ecid0 == ij[1])[0]
                    if len(a)>len(b):
                        vi,vj = a
                        vk = b[0]
                    else:
                        vi,vj = b
                        vk = a[0]
                    ei = tri_ij0[vi]
                    ej = tri_ij0[vj]
                    ek = tri_ij0[vk]
                    vt = self.xe[ej]-self.xe[ei]
                    mid1 = (self.xe[ei] + self.xe[ek]) / 2
                    mid2 = (self.xe[ej] + self.xe[ek]) / 2
                    if scaling:
                        mid1 = mid1*scale+deltax
                        mid2 = mid2*scale+deltax
                    self.contact_midedge_dict[ij].append([mid1, mid2])
                    self.contact_ten_dict[ij].append(h_int[tri_ij0].mean())

        print("add .contact_midedge_dict")
        print("add .contact_energy_dict")

    def contact_tension(self, scaling = True):
        """require sem.e_sigma, run sem.compute_tensor() first"""
        if self.subset_mode:
            e_sigma = self.sem.e_sigma[self._e_mask]
        else:
            e_sigma = self.sem.e_sigma
        
        deltax = self.sem.deltax
        scale = self.sem.scale
        self.contact_midedge_dict = defaultdict(list)
        self.contact_ten_dict = defaultdict(list)

        for ij in self.contact_tri_dict:
            tri_ij = self.simplices[self.contact_tri_dict[ij]]
            if len(ij)==2:
                for tri_ij0 in tri_ij:
                    ecid0 = self.ecid[tri_ij0]
                    a=np.where(ecid0 == ij[0])[0]
                    b=np.where(ecid0 == ij[1])[0]
                    if len(a)>len(b):
                        vi,vj = a
                        vk = b[0]
                    else:
                        vi,vj = b
                        vk = a[0]
                    ei = tri_ij0[vi]
                    ej = tri_ij0[vj]
                    ek = tri_ij0[vk]
                    vt = self.xe[ej]-self.xe[ei]
                    mid1 = (self.xe[ei] + self.xe[ek]) / 2
                    mid2 = (self.xe[ej] + self.xe[ek]) / 2
                    if scaling:
                        mid1 = mid1*scale+deltax
                        mid2 = mid2*scale+deltax
                    self.contact_midedge_dict[ij].append([mid1, mid2])
                    vt /= np.linalg.norm(vt)
                    c = vt[1]**2-vt[0]**2
                    d = 4*vt[0]*vt[1]
                    ten_i = 0
                    for j in [ei,ej,ek]:
                        sigma_j = e_sigma[j]
                        ten_i += c*(sigma_j[0,0]-sigma_j[1,1])-d*sigma_j[0,1]
                    self.contact_ten_dict[ij].append(ten_i/3)

        print("add .contact_midedge_dict")
        print("add .contact_ten_dict")
        
    # def pressure(self):
    
    def cell_tension(self):
        """require sem.e_sigma, run sem.compute_tensor() first"""
        if self.subset_mode:
            e_sigma = self.sem.e_sigma[self._e_mask]
        else:
            e_sigma = self.sem.e_sigma

        cid_list = [] # id of cells that are detected boundaries, corresponding to ten_list
        ten_list = [] # tension of each cell
        for i in range(self.nc):
            eid_ib = self.cell_boundary[i].copy()
            if len(eid_ib)>0:
                if eid_ib[-1]!=eid_ib[0]:
                    eid_ib.append(eid_ib[0])
                # xe_ib = self.xe[eid_ib]
                ten_i =  np.zeros(len(eid_ib)-1)
                eid_ib1 = np.roll(eid_ib[:-1],1)
                eid_ib2 = np.roll(eid_ib[:-1],-1)
                vt = self.xe[eid_ib1]-self.xe[eid_ib2]
                vt /= np.linalg.norm(vt,axis=1, keepdims=True)
                for n,j in enumerate(eid_ib[:-1]):# todo vectorize tension and sigma computation
                    sigma_j = e_sigma[j]
                    ten_i[n] = (vt[n,1]**2-vt[n,0]**2)*(sigma_j[0,0]-sigma_j[1,1])-4*vt[n,0]*vt[n,1]*sigma_j[0,1]
                # xe_b_rs = xe_ib.reshape(-1, 1, 2)
                ten_list.append(ten_i)
                cid_list.append(i)
        self.cell_ten_dict = dict(zip(cid_list,ten_list))#, xe_b, searching
        print("add .cell_ten_dict")

    def cell2contact_tension(self,scaling = True):
        """run cell_tension() first"""
        deltax = self.sem.deltax
        scale = self.sem.scale
        self.contact_midedge_dict1 = defaultdict(list)
        self.contact_ten_dict1 = defaultdict(list)
        for ij in self.contact_tri_dict:
            tri_ij = self.simplices[self.contact_tri_dict[ij]]
            if len(ij)==2:
                for tri_ij0 in tri_ij:
                    ecid0 = self.ecid[tri_ij0]
                    mid = []
                    ten_mid = []
                    for vi,vj in [(0,1), (1,2), (2,0)]:
                        ci = ecid0[vi]
                        cj = ecid0[vj]
                        if ci!=cj:
                            ei = tri_ij0[vi]
                            ej = tri_ij0[vj]
                            ten_ei_I = np.where(np.array(self.cell_boundary[ci]) == ei)[0]
                            ten_ej_I = np.where(np.array(self.cell_boundary[cj]) == ej)[0]
                            mid_ij = (self.xe[ei] + self.xe[ej]) / 2
                            if scaling:
                                mid_ij = mid_ij*scale+deltax
                            mid.append(mid_ij)
                            if len(ten_ei_I)>0 and len(ten_ej_I)>0:
                                ten_mid.append(self.cell_ten_dict[ci][ten_ei_I[0]] + self.cell_ten_dict[cj][ten_ej_I[0]])
                            else:
                                ten_mid.append(np.nan)
                    self.contact_ten_dict1[ij].append(ten_mid)
                    self.contact_midedge_dict1[ij].append(mid)
        print("add .contact_midedge_dict1")
        print("add .contact_ten_dict1")

class cellshape_3d():
    def __init__(self,vertices,indices):
        self.vertices = vertices
        self.indices = indices

class AlphaShape3():
    """
    3D Alpha-shape for element representation of cells.

    Attributes
    ----------
    vertices : np.ndarray
        Input points with shape (n_vertices, 3).
    alpha : Optional[float]
        Current alpha radius value.
    alpha_best : Optional[float]
        Optimized alpha radius value.
    simplices : np.ndarray
        Delaunay triangulation simplices.
    neighbors : np.ndarray
        Neighboring simplices indices.
    dmax : np.ndarray
        Maximum edge length of each simplex.
    elements : np.ndarray
        Unique elements in the triangulation.
    elements_counts : np.ndarray
        Occurrence count of each elements.
    tri2edge_I : np.ndarray
        Mapping from simplices to edges.
    alpha_range : list[float]
        Valid alpha range [min_edge, max_edge].
    simplices_adjacency_matrix : csr_matrix
        Adjacency matrix of simplices.
    is_boundary : np.ndarray
        Boolean array marking boundary edges.
    valid_simplices : np.ndarray
        Boolean array marking valid simplices.
    """
    
    vertices: NDArray
    """vertices coordinates"""
    n_vertices: int
    """Number of vertices"""
    simplices: NDArray
    """Delaunay triangulation simplices, shape=(n_simplices,4), dtype=np.int32"""
    neighbors: NDArray
    """Neighboring simplices indices, shape=(n_simplices,3), dtype=np.int32"""
    simplices_adjacency_matrix: csr_matrix
    """Simplices adjacency matrix, shape=(n_simplices,n_simplices)"""
    dmax: NDArray
    """Maximum edge length of each simplex, shape=(n_simplices)"""
    elements : NDArray
    """Unique elements in the triangulation"""
    elements_counts : NDArray
    """Occurrence count of each elements."""
    is_boundary : NDArray
    """Boolean array marking boundary elements"""
    valid_simplices : NDArray
    """Boolean array marking shape simplices"""
    alpha: float
    """Current alpha radius"""
    alpha_best: float
    """Optimized alpha radius"""
    vertices_data: Union[None, NDArray]

    def __init__(
            self,
            x: NDArray, 
            vertices_data: Optional[NDArray] = None,
            alpha: Optional[float] = None,
            no_hole: bool = True,
    ) -> None:
        self.vertices = x
        self.set_vertices_data(vertices_data)
        self.n_vertices = x.shape[0]
        tri = Delaunay(x) # perform Delaunay triangulation on the points
        self.simplices = tri.simplices # int32,
        self.neighbors = tri.neighbors # int32,
        self.dmax = np.array([max(pdist(x[simplex])) for simplex in self.simplices])
        # Get unique elements, their indices and counts
        simplices_sort = np.sort(self.simplices,axis=1)
        all_elements = np.concatenate([ 
            simplices_sort[:,[0,1,2]],
            simplices_sort[:,[0,1,3]], 
            simplices_sort[:,[0,2,3]], 
            simplices_sort[:,[1,2,3]] 
        ])# all elements in the Delaunay triangulation
        # unique elements in the Delaunay triangulation, their indices and counts
        self.elements, I, self.elements_counts = np.unique(all_elements,axis=0,return_inverse=True,return_counts=True)
        self.simplex2element_I = I.reshape(4,-1).T # map simplex to their elements

        # Simplices adjacency matrix
        n_simplices = len(self.simplices)
        simplex_indices, neighbor_offsets = np.where(self.neighbors!= -1) # method2 310 µs ± 201 ns per loop
        neighbor_indices = self.neighbors[simplex_indices, neighbor_offsets]
        rows = np.concatenate([simplex_indices, neighbor_indices])
        cols = np.concatenate([neighbor_indices, simplex_indices])
        coo_adj = csr_matrix((np.ones_like(rows), (rows, cols)), shape=(n_simplices, n_simplices)).tocoo()
        coo_adj.sum_duplicates()
        self.simplices_adjacency_matrix = coo_adj.tocsr()
        
        # Range of alpha radius
        self.alpha_range = [np.min(self.dmax),np.max(self.dmax)]
        self.alpha = alpha
        self.alpha_best = None
        if self.alpha is None:
            self.optimize_alpha(no_hole)
            self.alpha = self.alpha_best
        # self.alpha = self.alpha_range[1]
        self.update(self.alpha)

    def __repr__(self) -> str:
        return f'3D alpha-shape with alpha = {self.alpha}, {len(self.simplices)} simplices, {self.n_vertices} points'
    
    def update(self, alpha: float) -> None:
        """
        Update the alpha shape based on the given alpha radius. Calculate valid simplices and boundary edges.
        
        Parameters
        ------
        alpha : float
            Alpha radius used to determine valid simplices and boundaries.
        """
        self.alpha = alpha
        # Determine boundary edges, whose counts = 1
        edge_counts_update = self.elements_counts.copy()
        for i in self.simplex2element_I[self.dmax>=alpha].flatten():
            edge_counts_update[i] -= 1
        self.is_boundary = edge_counts_update == 1
        # Mark simplices (maximum edge length< alpha radius) as valid
        self.valid_simplices = self.dmax<=alpha

    def optimize_alpha(
            self, 
            no_hole: bool = True, 
            return_alpha: bool = False
    ) -> Union[float, None]:
        alpha_low, alpha_high = self.alpha_range[0], self.alpha_range[1]
        alpha_best = alpha_high
        tol = 1e-3
        max_iterations = 10
        for _ in range(max_iterations):
            alpha_mid = (alpha_low + alpha_high) / 2
            self.update(alpha_mid)  # Update valid_simplices and is_boundary

            # Check if all points are covered
            cover_points = np.unique(self.simplices[self.valid_simplices].flatten())
            all_covered = cover_points.shape[0] == self.n_vertices
            # Check if all simplices connected
            adjacency_matrix = self.simplices_adjacency_matrix[self.valid_simplices,:][:,self.valid_simplices]
            order = breadth_first_order(
                    csgraph=adjacency_matrix,
                    i_start=0,
                    directed=False,
                    return_predecessors=False
                )
            all_connected = len(order) == adjacency_matrix.shape[0]
            if no_hole:
                # Require exactly one boundary loop and all points covered
                # condition_met = all_covered and all_connected and (len(boundaries) == 1)
                boundary_elements = self.get_boundary_elements()
                chi = compute_euler_characteristic(boundary_elements)[0]
                condition_met = all_covered and all_connected and chi==2
            else:
                # Original logic (fallback, adjust if necessary)
                # Here, we just ensure all points are covered (no hole check)
                condition_met = all_covered and all_connected

            if condition_met:
                alpha_best = alpha_mid
                alpha_high = alpha_mid
            else:
                alpha_low = alpha_mid

            if alpha_high - alpha_low < tol:
                break

        self.alpha_best = alpha_best
        if return_alpha:
            return alpha_best

    def get_simplices(self) -> NDArray:
        return self.simplices[self.valid_simplices]
    
    def get_boundary_elements(self) -> NDArray:
        return self.elements[self.is_boundary]
    
    def set_vertices_data(self, vertices_data):
        self.vertices_data = vertices_data

def compute_euler_characteristic(faces):
    V = len(np.unique(faces.flatten()))
    F = faces.shape[0]
    simplices_sort = np.sort(faces,axis=1)
    all_elements = np.concatenate([ 
        simplices_sort[:,[0,1]],
        simplices_sort[:,[0,2]], 
        simplices_sort[:,[1,2]] 
    ])# all elements in the Delaunay triangulation
    unique_edges = np.unique(all_elements,axis=0)
    E = unique_edges.shape[0]
    chi = V - E + F
    return chi,V,E,F, 