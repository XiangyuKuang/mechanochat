from typing import Optional, Union, Tuple, Dict, List, Iterable, Literal, Sequence, overload
from anndata import AnnData
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib.typing import ColorType
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, to_rgb, to_hex, LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch, PathPatch, FancyArrowPatch, Arc, ArrowStyle, Rectangle
import matplotlib.patheffects as path_effects
from matplotlib.transforms import offset_copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict
from scipy.spatial import ConvexHull
from scipy.sparse import csr_matrix, csr_array, isspmatrix
from sklearn.preprocessing import minmax_scale
from math import pi
from ..SEM import SEM2, SEM3
from .._utils import cell_tri_dev, AlphaShape
import pyvista as pv
from tqdm import tqdm

def alphashape_plot3(
    sem: SEM3,
    vis_key: Optional[str] = None,
    arr: Optional[Union[NDArray, pd.Series]] = None,
    summary: str = 'receiver_mechano',
    cid_list: Optional[NDArray] = None, 
    cmap_name: str = 'Reds',
    palette: Optional[Dict[str,Tuple]] = None,
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
    boundary_alpha: float = 1,
    face_alpha: Union[float,List,Tuple] = 1,
    face_alpha_arr: Optional[NDArray] = None,
    smooth_shape: bool = False,
    plotter: Union[pv.Plotter,None] = None,
) -> pv.Plotter:
    
    cid_list = _get_cid_list(cid_list, sem.nc)
    arr = _get_arr(sem, vis_key, arr, summary)
    
    if arr is None:
        # vis sem.ctype
        if vis_key is None:
            # use cell type color in sem
            cat_code = sem.ctype[cid_list]
            cat_list = sem.ctype_list
            color_list = sem.color_list
            colors = color_list[cat_code]
            facecolors = np.insert(colors,3,face_alpha,axis=1)
            edgecolors = np.insert(colors,3,boundary_alpha,axis=1)
            enable_colorbar = False
        else:
            raise KeyError(f"vis_key '{vis_key}' not found in genes or adata.obs")
    else:
        # vis arr
        if arr.dtype.name == 'category':
            # obtain category and color from arr
            cat_code, cat_list, color_list = _get_cat_arr_color(sem,arr,cid_list,vis_key,cmap_name,palette)
            colors = color_list[cat_code]
            facecolors = np.insert(colors,3,face_alpha,axis=1)
            edgecolors = np.insert(colors,3,boundary_alpha,axis=1)
            enable_colorbar = False
        else:
            # set color based on arr
            if len(arr) == sem.nc:
                arr = arr[cid_list]
            elif len(arr)!=len(cid_list):
                raise ValueError('len(arr)!=len(cid_list)')
            
            cmap = colormaps[cmap_name]
            # vmax = np.percentile(arr,95) if vmax is None else vmax
            vmax = arr.max() if vmax is None else vmax
            vmin = arr.min() if vmin is None else vmin
            norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
            facecolors = cmap(norm(arr))
            edgecolors = cmap(norm(arr))
            if isinstance(face_alpha, (list, tuple)):
                face_alpha = minmax_scale(arr, tuple(face_alpha))
            facecolors[:,3] = face_alpha
            edgecolors[:,3] = boundary_alpha
            enable_colorbar = True
            # if enable_legend:
            #     print('visualize data, cannot use legend')
            enable_legend = False
    if face_alpha_arr is not None:
        facecolors[:,3] = face_alpha_arr
    # draw cell shape
    if not smooth_shape:
        # --- 快速版本：循环外一次性构建 PolyData，不做细分/平滑，按面赋色 ---
        all_vertices = []
        all_faces  = []
        all_colors = []
        m = 0
        for n, cid in enumerate(cid_list):
            indices = sem.shp_list[cid].get_boundary_elements() + m
            faces = np.insert(indices,0,3,axis=1)
            vertices = sem.shp_list[cid].vertices
            all_vertices.append(vertices)
            all_faces.append(faces)
            all_colors.append(np.tile(facecolors[n],(indices.shape[0],1)))  # 按面赋色
            m += vertices.shape[0]
        all_vertices = np.vstack(all_vertices)               # (sumNi, 3)
        all_faces  = np.vstack(all_faces)                # (sumMi, 3)
        all_colors   = np.vstack(all_colors)
        mesh = pv.PolyData(all_vertices, faces=all_faces)
        mesh.cell_data['colors'] = all_colors
    else:
        # --- smooth 版本：逐细胞 clean+subdivide+smooth_taubin，颜色赋在最终顶点上 ---
        blocks = []
        for n, cid in enumerate(tqdm(cid_list,"Smoothing Cell Shapes")):
            indices = sem.shp_list[cid].get_boundary_elements()
            if indices.shape[0] == 0:                            # 无边界面的细胞，跳过
                continue
            faces = np.insert(indices,0,3,axis=1)
            cell = pv.PolyData(sem.shp_list[cid].vertices, faces=faces)
            cell = cell.clean()                                  # 移除未引用点/合并重合点
            if cell.n_points == 0:
                continue
            # subdivide 要求流形网格；α-shape 边界可能非流形(一条边被>2个面共享)，
            # 非流形细胞只做 Taubin 平滑(不要求流形)，避免触发 vtk 报错刷屏
            nm = cell.extract_feature_edges(
                boundary_edges=False, feature_edges=False,
                manifold_edges=False, non_manifold_edges=True,
            )
            if nm.n_cells == 0:
                cell = cell.subdivide(2,subfilter = 'butterfly')                         # 细分 + 平滑
            cell = cell.smooth_taubin(n_iter=20, pass_band=0.05) # 非收缩平滑 n_iter=20, pass_band=0.1
            if cell.n_points == 0:                               # 平滑后退化为空，跳过
                continue
            # 颜色赋在最终顶点，避免 subdivide 丢失 data array
            cell.point_data['colors'] = np.tile(facecolors[n], (cell.n_points, 1))
            blocks.append(cell)
        mesh = blocks[0].merge(blocks[1:]) if len(blocks) > 1 else blocks[0]
    
    if plotter is None:
        pv.set_jupyter_backend('static')
        plotter = pv.Plotter(notebook=True, window_size=[1920, 1080])
    
    plotter.add_mesh(mesh,
        scalars='colors',
        rgb=True,
        show_edges=False,
        smooth_shading=smooth_shape,
        backface_culling=False,
        # specular=1,
        ambient=0.3,
        # lighting = True
    )

    return plotter

def alphashape_plot(
    sem: SEM2, 
    vis_key: Optional[str] = None,
    obsm_key: Optional[str] = None,
    arr: Optional[Union[NDArray, pd.Series]] = None, 
    summary: str = 'receiver_mechano_signal',
    compute_alphashape: bool = False, 
    cid_list: Union[NDArray[np.int_], NDArray[np.bool_], Sequence[int], None] = None, 
    cmap_name: str = 'Reds',
    palette: Optional[Dict[str,Tuple]] = None,
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
    boundary_width: float = 0.5, 
    boundary_color: Optional[ColorType] = 'gray', 
    boundary_alpha: float = 1, 
    face_color: Union[str, NDArray[np.floating], Sequence[float], None] = None,
    face_alpha: Union[float,Iterable[float]] = 1,
    show_axis: bool = False,
    enable_annotation: bool= False,
    enable_legend: bool = True,
    enable_colorbar: bool = True,
    return_mappable: bool = False,
    ax: Optional[Axes] = None, 
    show: bool = True,
    save_name: Optional[str] = None,
    rotation: Optional[Dict] = None,
    **kwargs
):
    """
    Plot cell shape using alpha shape, visualize cell data by colors.

    Parameters
    ----------
    sem : SEM
        Subcellular element method object
    vis_key : str, optional
        Key to retrieve visualization data from `sem.adata`
    arr : np.ndarray or pd.Series, optional
        Data for visualization. shape = (nc,) or (len(cid_list),)
        
        Ignored if `vis_key` is provided .
        
        `sem.ctype` will be visualized if `arr` and `vis_key` both are not provided.
    summary : str
        if `summary == 'gene'`, retrieve data from `adata[:, vis_key]`

        if `summary == 'cell'`, retrieve data from `adata.obs[vis_key]`

        if `summary in adata.obsm`, retrieve data from `adata.obsm[summary][vis_key]`
    compute_alphashape : bool, default=False
        Compute alphashape if True
    cid_list : ndarray, optional
        Array of index for cells to be visualized. Default: all cells
    cmap_name : str, default='Reds'
        Valid matplotlib colormap name to visualize data
    vmax : float, optional
        Colormap upper bound. Default: 95th percentile for positive data
    vmin : float, optional
        Colormap lower bound. Default: data min
    boundary_width : float, default=1
        Cell boundary line width
    boundary_color : str or tuple, optional
        Cell boundary line color, Default: matches face color
    boundary_alpha : float, default=1
        Cell boundary line opacity, 0 (fully transparent), 1 (fully opaque)
    face_alpha : float, default=1
        Cell shape face opacity, 0 (fully transparent), 1 (fully opaque)
    show_axis : bool, default=True
        Show axis
    enable_annotation : bool, default=False
        Annotate cells with index at centroids
    enable_legend : bool, default=False
        Show categorical legend (only for category data)
    enable_colorbar : bool, default=False
        Show colorbar (only for continuous data)
    ax : Axes, optional
        Target matplotlib axes object. Creates new figure if None.
    save_name : str, optional
        Output path for figure saving (e.g., 'figure.pdf')
    **kwargs
        keyword arguments passed to `sem.compute_alphashape()`
        
    Returns
    ----------
    ax : Axes
    """

    if compute_alphashape or not sem.alphashape_info['computed'] or kwargs:
        sem.compute_alphashape(**kwargs)

    fig, ax = _get_axes(ax)
    cid_list = _get_cid_list(cid_list, sem.nc)
    # arr = _get_arr(sem, vis_key, arr, summary)
    arr = _get_arr1(sem, vis_key, arr, summary, obsm_key)
    
    if arr is None:
        # vis sem.ctype
        if vis_key is None:
            # use cell type color in sem or face_color
            cat_code = sem.ctype[cid_list]
            cat_list = sem.ctype_list
            # set color_list by palette
            if palette is None:
                color_list = sem.color_list
            else:
                color_list = [palette[x] for x in cat_list]
                # if type(color_list[0]) is str:
                #     color_list = [to_rgb(x) for x in color_list]
                # color_list = np.array(color_list)
                color_list = np.array([to_rgb(x) for x in color_list])

            if face_color is None:
                # set face color by color_list
                colors = color_list[cat_code]
            else:
                # set face color by face_color
                if type(face_color) is str:
                    face_color = to_rgb(face_color)
                    colors = np.tile(np.array(face_color),[sem.nc,1])
                else:
                    colors = face_color
                enable_legend = False
            
            facecolors = np.insert(colors,3,face_alpha,axis=1)
            edgecolors = np.insert(colors,3,boundary_alpha,axis=1)
            enable_colorbar = False
        else:
            raise KeyError(f"vis_key '{vis_key}' not found in '{summary}'")
    else:
        # vis arr
        if arr.dtype.name == 'category':
            # obtain category and color from arr
            cat_code, cat_list, color_list = _get_cat_arr_color(sem,arr,cid_list,vis_key,cmap_name,palette)
            colors = color_list[cat_code]
            if isinstance(face_alpha, (list, tuple)):
                face_alpha = minmax_scale(arr, tuple(face_alpha))
            if isinstance(boundary_alpha, (list, tuple)):
                boundary_alpha = minmax_scale(arr, tuple(boundary_alpha))
            facecolors = np.insert(colors,3,face_alpha,axis=1)
            edgecolors = np.insert(colors,3,boundary_alpha,axis=1)
            enable_colorbar = False
        else:
            # set color based on arr
            if len(arr) == sem.nc:
                arr = arr[cid_list]
            elif len(arr)!=len(cid_list):
                raise ValueError('len(arr)!=len(cid_list)')
            
            if isinstance(face_alpha, (list, tuple)):
                face_alpha = minmax_scale(arr, tuple(face_alpha))
            # if isinstance(boundary_alpha, (list, tuple)):
            #     boundary_alpha = minmax_scale(arr, tuple(boundary_alpha))

            if summary == 'cell':
                cat_code = sem.ctype[cid_list]
                cat_list = sem.ctype_list
                color_list = sem.color_list
                colors = color_list[cat_code]
                enable_colorbar = False
                facecolors = np.insert(colors,3,face_alpha,axis=1)
                edgecolors = np.insert(colors,3,boundary_alpha,axis=1)
            else:
                cmap = colormaps[cmap_name]
                # vmax = np.percentile(arr,95) if vmax is None else vmax
                vmax = arr.max() if vmax is None else vmax
                vmin = arr.min() if vmin is None else vmin
                norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
                facecolors = cmap(norm(arr))
                edgecolors = cmap(norm(arr))
                facecolors[:,3] = face_alpha
                edgecolors[:,3] = boundary_alpha
                enable_legend = False
                # enable_colorbar = True
                # if enable_legend:
                #     print('visualize data, cannot use legend')

    if rotation is not None:
        center = rotation['center']
        theta = rotation['theta']
        RT = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).T
        enable_rotation = True
    else:
        enable_rotation = False

    # draw cell shape
    all_boundaries = []
    fc = []
    bc = []
    for i, cid in enumerate(cid_list):
        b = sem.alphashape[cid].get_boundary()
        if enable_rotation:
            b = (b-center) @ RT + center
        all_boundaries.append(b)
        fc.append(facecolors[i])
        bc.append(edgecolors[i])
    fc = np.vstack(fc)
    bc = np.vstack(bc)
    if boundary_color is not None: # set edgecolors by arg `boundary_color`
        bc = np.insert(np.array(to_rgb(boundary_color)),3,boundary_alpha)
    
    polyc = PolyCollection(
        all_boundaries,
        facecolors=fc,
        edgecolors=bc,
        linewidths = boundary_width,
        # antialiased = True,
        path_effects=[path_effects.Stroke(capstyle="round",joinstyle="round")],
    )
    ax.add_collection(polyc)
    _set_axes(ax, show_axis)
    
    if enable_colorbar:
        # draw colorbar
        _add_colorbar(fig, ax, cmap, norm)
    elif enable_legend:
        # draw legend
        legend_patches = []
        for i in np.unique(cat_code):
            legend_patches.append(Patch(color=color_list[i],label=cat_list[i]))
        transform = offset_copy(ax.transAxes, x=5, y=0, units='points',fig=fig) 
        ax.legend(handles=legend_patches,
                  loc='center left',
                  bbox_to_anchor=(1, 0.5),
                  bbox_transform=transform,
                  frameon=False)
    
    if enable_annotation:
        spatial_coor = sem.xc*sem.scale+sem.deltax
        if rotation:
            spatial_coor = (spatial_coor-center) @ RT + center
        for i in cid_list:
            ax.annotate(f'{i}',spatial_coor[i],ha='center',va='center',fontsize=8)#fontweight='bold',font

    _save_close(fig,save_name,show)

    if return_mappable:
        if 'cmap' in locals() and 'norm' in locals():
            return ax, cmap, norm
        else:
            return ax, None, None
    return ax

def element_plot(
    sem: SEM2, 
    vis_key: Optional[str] = None,
    arr: Optional[Union[NDArray, pd.Series]] = None,
    summary: str = 'sender',
    cid_list: Union[NDArray[np.int_], NDArray[np.bool_], Sequence[int], None] = None, 
    cmap_name: str ='Reds', 
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
    spot_size: float = 1,
    scaling: bool = True, 
    show_axis: bool = True, 
    enable_colorbar: bool = True, 
    enable_legend: bool = True,
    ax: Optional[Axes] = None,
    save_name: Optional[str] = None,
    show: bool = True,
    rotation: Optional[Dict] = None,
) -> Axes:
    """
    Plotting cell elements

    Parameters
    ----------
    sem : SEM
        Subcellular element method object
    vis_key : str, optional
        Key to retrieve visualization data from `sem.adata`.
    arr : np.ndarray or pd.Series, optional
        Data for visualization. Accepts both cell-level (nc,) and element-level (ne,)
    summary : str, default='sender'
        'sender' represents sender signal, retrieves data from adata.obsm['sender_signal'][vis_key]

        'receiver' retrieves receiver signal data from adata.obsm['receiver_signal'][vis_key]

        'gene' retrieves gene expression data from adata
    cid_list : ndarray, optional
        Array of index for cells to be visualized. Default: all cells
    cmap_name : str, default='Reds'
        Valid matplotlib colormap name to visualize data
    spot_size : float, default=1
        Markersize for `matplotlib.pyplot.scatter`
    scaling : bool, default=True
        Scale coordinates back to original data(`xc`) if True, otherwise visualize directly.
    show_axis : bool, default=True
        Show axis.
    enable_legend : bool, default=False
        Show categorical legend (only for category data).
    enable_colorbar : bool, default=False
        Show colorbar (only for continuous data).
    ax : Axes, optional
        Target matplotlib axes object. Creates new figure if None
    save_name : str, optional
        Output path for figure saving (e.g., 'figure.pdf')
    
    Returns
    ----------
    ax : Axes
    """

    fig, ax = _get_axes(ax)
    # cid_list, xe = _get_cid_list(sem, cid_list, scaling=scaling, element_plot=True)
    cid_list = _get_cid_list(cid_list, sem.nc)
    arr = _get_arr(sem, vis_key, arr, summary)

    if rotation is not None:
        center = rotation['center']
        theta = rotation['theta']
        RT = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).T
        enable_rotation = True
    else:
        enable_rotation = False

    ec = None
    if arr is None:
        # vis sem.ctype
        if vis_key is None:
            # use cell type color in sem
            cat_code = sem.ctype[cid_list]
            cat_list = sem.ctype_list
            color_list = sem.color_list
        else:
            raise KeyError(f"vis_key '{vis_key}' not found in genes or adata.obs")
    else:
        # vis arr
        if arr.dtype.name == 'category':
            # obtain category and color from arr
            cat_code, cat_list, color_list = _get_cat_arr_color(sem,arr,cid_list,vis_key,cmap_name)
        else:
            cmap = colormaps[cmap_name]
            # color norm
            vmax = arr.max() if vmax is None else vmax
            vmin = arr.min() if vmin is None else vmin
            norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
            # if arr.min()>=0:
            #     norm = Normalize(vmin=arr.min(), vmax=np.percentile(arr,95), clip=False)
            # else:
            #     a = np.percentile(np.abs(arr),95)
            #     norm = Normalize(vmin=-a, vmax=a, clip=False)
            # set color
            if arr.shape[0] == len(cid_list):
                # cell color
                cc = cmap(norm(arr))
                # cell color -> element color
                ec = np.zeros((sem.ne,cc.shape[1]))
                for cid in range(sem.nc):
                    ne_i = sem.ceidn[cid+1]-sem.ceidn[cid]
                    ec[sem.ceidn[cid]:sem.ceidn[cid+1],:] = np.tile(cc[cid],(ne_i,1))
            else:
                ec = cmap(norm(arr)) # element color
    # plot
    if ec is None:
        xe = _get_xe(sem, cid_list, scaling)
        if enable_rotation:
            xe = (xe-center) @ RT + center
        # cell color
        ecid = []
        for n,cid in enumerate(cid_list):
            ecid.append(n*np.ones(sem.ceidn[cid+1]-sem.ceidn[cid]))
        ecid = np.concatenate(ecid).astype(int)
        element_cat = cat_code[ecid]
        for i in np.unique(cat_code):
            vis = element_cat == i
            if sem.dim == 3:
                ax.scatter(
                    xe[vis, 0], xe[vis, 1], xe[vis, 2],
                    c = color_list[i][np.newaxis],
                    label=cat_list[i],
                    s=spot_size
                )
            else:
                ax.scatter(
                    xe[vis, 0], xe[vis, 1],
                    c = color_list[i][np.newaxis],
                    label=cat_list[i],
                    s=spot_size
                )
        if enable_legend:
            # draw legend
            transform = offset_copy(ax.transAxes, x=5, y=0, units='points',fig=fig) 
            ax.legend(loc='center left',
                      bbox_to_anchor=(1, 0.5),
                      bbox_transform=transform,
                      frameon=False,
                      markerscale=5/spot_size)
    else:
        # element color
        xe = sem.xe*sem.scale+sem.deltax if scaling else sem.xe
        if enable_rotation:
            xe = (xe-center) @ RT + center
        for cid in cid_list:
            if sem.dim == 3:
                ax.scatter(
                    xe[sem.ceidn[cid]:sem.ceidn[cid+1], 0],
                    xe[sem.ceidn[cid]:sem.ceidn[cid+1], 1],
                    xe[sem.ceidn[cid]:sem.ceidn[cid+1], 2], 
                    c = ec[sem.ceidn[cid]:sem.ceidn[cid+1]],
                    s=spot_size
                )
            else:
                ax.scatter(
                    xe[sem.ceidn[cid]:sem.ceidn[cid+1], 0],
                    xe[sem.ceidn[cid]:sem.ceidn[cid+1], 1], 
                    c = ec[sem.ceidn[cid]:sem.ceidn[cid+1]],
                    s=spot_size
                )
        if enable_colorbar:
            # draw colorbar
            _add_colorbar(fig, ax, cmap, norm)
    _set_axes(ax, show_axis)
    _save_close(fig,save_name,show)
    return ax

def forceij_plot(
    sem: SEM2,
    cid_i: Iterable[int],
    cid_j: Union[None,Iterable[int]] = None,
    fx_csr: Union[None,csr_array] = None,
    fy_csr: Union[None,csr_array] = None,
    ax: Optional[Axes] = None,
    reciprocal: bool = True,
    width_bin = None,
    # fmax: float = 1,
    fmin: float = 1e-1,
    rotation: Optional[Dict] = None,
    **quiver_kwargs
) -> Axes:
    """
    Plot intercellular forces between elements of selected cells using quiver arrows.

    For each element in `cid_i`, computes the total force exerted by elements in `cid_j`
    and renders it as a quiver arrow. Arrow length is scaled by log(1 + force magnitude).

    Parameters
    ----------
    sem : SEM1 or SEM2
        Subcellular element method object
    cid_i : iterable of int
        Cell indices acting as force receivers. When `cid_j` is None, each cell in
        `cid_i` is treated as receiver against all remaining cells in `cid_i`.
    cid_j : iterable of int, optional
        Cell indices acting as force senders. If None, pairwise forces within `cid_i`
        are visualized (each cell vs. all others in the list).
    fx_csr : csr_array, optional
        Precomputed x-component force sparse matrix. Computed from `sem` if not provided.
    fy_csr : csr_array, optional
        Precomputed y-component force sparse matrix. Computed from `sem` if not provided.
    ax : Axes, optional
        Target matplotlib axes object. Creates new figure if None.
    reciprocal : bool, default=True
        If True and `cid_j` is provided, also plot the reciprocal forces from `cid_i`
        elements back onto `cid_j` elements.
    width_bin : array-like, optional
        Force magnitude bin boundaries for variable arrow width. If None, a single
        arrow width is used scaled by log(1 + force magnitude).
    fmax : float, default=1
        Upper bound for force magnitude scaling (currently unused in log scaling mode).
    fmin : float, default=1e-1
        Minimum force magnitude threshold; elements with force below this value are hidden.
    **quiver_kwargs
        Additional keyword arguments passed to `ax.quiver()`.

    Returns
    -------
    ax : Axes
    """
    fig, ax = _get_axes(ax)

    quiver_default = {
        'angles':'xy',
        'headwidth':3.5,
        'headlength': 4,
        'headaxislength':3.5,
        'width':0.01,
        # 'minshaft':1.2,
        'scale':20
        } #,'width':0.002,'headlength':5
    quiver_default.update(quiver_kwargs)

    if type(sem) is SEM2:
        cid_list = np.asarray(cid_i)
        if cid_j is not None:
            cid_list = np.concatenate( (cid_list, np.asarray(cid_j)) )
        e_mask = np.isin(sem.ecid, cid_list)

        xe = sem.xe[e_mask]*sem.scale+sem.deltax
        if rotation is not None:
            center = rotation['center']
            theta = rotation['theta']
            RT = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).T
            xe = (xe-center) @ RT + center
        
        if fx_csr is None or fy_csr is None:
            fx_csr, fy_csr = sem.f_pairs_tocsr(e_mask)
        else:
            fx_csr = fx_csr[e_mask,:][:,e_mask]
            fy_csr = fy_csr[e_mask,:][:,e_mask]

        if width_bin is not None:
            w = quiver_default['width']*np.linspace(1,2,len(width_bin))

        ecid = sem.ecid[e_mask]
        cid_i = list(cid_i)
        if cid_j is None:
            for i in range(len(cid_i)):
                cid_j = cid_i.copy()
                center_i = cid_i[i]
                cid_j.pop(i)

                eid_i = np.where(ecid==center_i)[0]
                I_eid_j = np.zeros_like(ecid,dtype=bool)
                for cid in cid_j:
                    I_eid_j = I_eid_j | (ecid==cid)
                eid_j = np.where(I_eid_j)[0]

                fx = fx_csr[eid_i,:][:,eid_j].sum(axis=1).A1
                fy = fy_csr[eid_i,:][:,eid_j].sum(axis=1).A1
                if rotation is not None:
                    center = rotation['center']
                    theta = rotation['theta']
                    RT = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).T
                    fvec = np.stack([fx, fy], axis=1) @ RT
                    fx, fy = fvec[:, 0], fvec[:, 1]
                
                fa = np.sqrt(fx**2+fy**2)
                fx = fx/fa
                fy = fy/fa
                I = fa>fmin
                # a = (np.clip(fa,fmin,fmax)-fmin)/(fmax-fmin)*0.6+0.5
                if width_bin is None:
                    a = np.log1p(fa)
                    fx = fx*a
                    fy = fy*a
                    ax.quiver(xe[eid_i[I],0],xe[eid_i[I],1],fx[I],fy[I],**quiver_default)
                else:
                    # pct_values = np.percentile(fa, width_pcts)
                    for i in range(len(width_bin)):
                        if i<len(width_bin)-1:
                            Iw = (fa>width_bin[i]) & (fa<=width_bin[i+1]) & I
                        else:
                            Iw = (fa>width_bin[i]) & I
                        quiver_default['width'] = w[i]
                        ax.quiver(
                            xe[eid_i[Iw],0],xe[eid_i[Iw],1],
                            fx[Iw],fy[Iw],
                            **quiver_default
                        )
        else:
            cid_j = list(cid_j)
            I_eid_i = np.zeros_like(ecid,dtype=bool)
            for cid in cid_i:
                I_eid_i = I_eid_i | (ecid==cid)
            eid_i = np.where(I_eid_i)[0]

            I_eid_j = np.zeros_like(ecid,dtype=bool)
            for cid in cid_j:
                I_eid_j = I_eid_j | (ecid==cid)
            eid_j = np.where(I_eid_j)[0]
            
            fx = fx_csr[eid_i,:][:,eid_j].sum(axis=1).A1
            fy = fy_csr[eid_i,:][:,eid_j].sum(axis=1).A1
            if rotation is not None:
                center = rotation['center']
                theta = rotation['theta']
                RT = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).T
                fvec = np.stack([fx, fy], axis=1) @ RT
                fx, fy = fvec[:, 0], fvec[:, 1]

            fa = np.sqrt(fx**2+fy**2)
            I = fa>fmin
            # a = (np.clip(fa,fmin,fmax)-fmin)/(fmax-fmin)*0.6+0.5
            a = np.log1p(fa)
            fx = fx/fa*a
            fy = fy/fa*a
            xe = sem.xe*sem.scale+sem.deltax
            if rotation is not None:
                center = rotation['center']
                theta = rotation['theta']
                RT = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).T
                xe = (xe-center) @ RT + center
            print(fx[I])
            fig,ax = plt.subplots()
            ax.quiver(xe[eid_i[I],0],xe[eid_i[I],1],fx[I],fy[I],**quiver_default)

            if reciprocal:
                fx_j = fx_csr[eid_j,:][:,eid_i].sum(axis=1).A1
                fy_j = fy_csr[eid_j,:][:,eid_i].sum(axis=1).A1
                fa = np.sqrt(fx_j**2+fy_j**2)
                I = fa>fmin
                # a = (np.clip(fa,fmin,fmax)-fmin)/(fmax-fmin)*0.6+0.5
                a = np.log1p(fa)
                fx_j = fx_j/fa*a
                fy_j = fy_j/fa*a
                ax.quiver(xe[eid_j[I],0],xe[eid_j[I],1],fx_j[I],fy_j[I],**quiver_default)
    else:
        if fx_csr is None or fy_csr is None:
            fx_csr, fy_csr = sem.f_pairs_tocsr() #fx_csr[i, j] = fx_{j->i}

        cid_i = list(cid_i)
        if cid_j is None:
            for i in range(len(cid_i)):
                cid_j = cid_i.copy()
                center_i = cid_i[i]
                cid_j.pop(i)

                eid_i = np.arange(sem.ceidn[center_i],sem.ceidn[center_i+1])

                eid_j = []
                for cid in cid_j:
                    eid_j.append(np.arange(sem.ceidn[cid],sem.ceidn[cid+1]))
                eid_j = np.concatenate(eid_j)

                fx = fx_csr[eid_i,:][:,eid_j].sum(axis=1).A1
                fy = fy_csr[eid_i,:][:,eid_j].sum(axis=1).A1
                fa = np.sqrt(fx**2+fy**2)
                I = fa>fmin
                # a = (np.clip(fa,fmin,fmax)-fmin)/(fmax-fmin)*0.6+0.5
                a = np.log1p(fa)
                fx = fx/fa*a
                fy = fy/fa*a
                xe = sem.xe*sem.scale+sem.deltax
                ax.quiver(xe[eid_i[I],0],xe[eid_i[I],1],fx[I],fy[I],**quiver_default)
        else:
            cid_j = list(cid_j)
            eid_i = []
            for cid in cid_i:
                eid_i.append(np.arange(sem.ceidn[cid],sem.ceidn[cid+1]))
            eid_i = np.concatenate(eid_i)

            eid_j = []
            for cid in cid_j:
                eid_j.append(np.arange(sem.ceidn[cid],sem.ceidn[cid+1]))
            eid_j = np.concatenate(eid_j)
            
            fx = fx_csr[eid_i,:][:,eid_j].sum(axis=1).A1
            fy = fy_csr[eid_i,:][:,eid_j].sum(axis=1).A1
            fa = np.sqrt(fx**2+fy**2)
            I = fa>fmin
            # a = (np.clip(fa,fmin,fmax)-fmin)/(fmax-fmin)*0.6+0.5
            a = np.log1p(fa)
            fx = fx/fa*a
            fy = fy/fa*a
            xe = sem.xe*sem.scale+sem.deltax
            ax.quiver(xe[eid_i[I],0],xe[eid_i[I],1],fx[I],fy[I],**quiver_default)

            if reciprocal:
                fx_j = fx_csr[eid_j,:][:,eid_i].sum(axis=1).A1
                fy_j = fy_csr[eid_j,:][:,eid_i].sum(axis=1).A1
                fa = np.sqrt(fx_j**2+fy_j**2)
                I = fa>fmin
                # a = (np.clip(fa,fmin,fmax)-fmin)/(fmax-fmin)*0.6+0.5
                a = np.log1p(fa)
                fx_j = fx_j/fa*a
                fy_j = fy_j/fa*a
                ax.quiver(xe[eid_j[I],0],xe[eid_j[I],1],fx_j[I],fy_j[I],**quiver_default)

    return ax

def signal_direction_plot3(
    sem: Optional[Union[SEM2,SEM3]] = None,
    adata: Optional[AnnData] = None,
    spatial_key: Optional[str] = 'spatial',
    sig_mat: Optional[Union[csr_matrix, NDArray]] = None,
    signal: Optional[str] = None,
    th: float = 0.,
    quiver_length: float = 0.45,
    quiver2d_param: dict = {},
    width_bin_n = 3,
    width_scale = [0.5,2],
    cid_list: Union[NDArray[np.int_], NDArray[np.bool_], Sequence[int], None] = None,
    scaling: bool = True,
    ax: Optional[Axes] = None,
    plotter: Optional[pv.Plotter] = None
) -> Union[Axes, pv.Plotter]:
    """
    Plot spatial directions of cell-cell communication using quiver arrows.

    For each sender-receiver cell pair with signal strength above `th`, draws an arrow
    from the sender centroid toward the receiver centroid. Arrow width is stratified by
    signal strength percentiles to convey relative signal intensity.

    Parameters
    ----------
    sem : SEM, SEM1, SEM2 or SEM3, optional
        Subcellular element method object. If provided, cell coordinates and signal
        matrix are retrieved from `sem`. Takes precedence over `adata`.
    adata : AnnData, optional
        AnnData object used when `sem` is None. Must contain spatial coordinates in
        `adata.obsm[spatial_key]` and the signal matrix in `adata.obsp[signal]`.
    spatial_key : str, default='spatial'
        Key in `adata.obsm` for spatial coordinates. Only used when `sem` is None.
    sig_mat : csr_matrix or ndarray, optional
        Signal matrix of shape `(nc, nc)`. `sig_mat[i, j]` is the signal strength
        from sender `j` to receiver `i`. Required if `signal` is not provided.
    signal : str, optional
        Key in `sem.adata.obsp` or `adata.obsp` to retrieve the signal matrix.
        Takes precedence over `sig_mat`.
    th : float or callable, default=0.
        Signal strength threshold. Sender-receiver pairs with signal <= `th` are not
        plotted. If callable, it is applied to all signal data values to compute `th`.
    quiver_length : float, default=0.45
        Fraction of the sender-to-receiver vector used as arrow length.
        1.0 means the arrow tip reaches the receiver centroid.
    quiver2d_param : dict, default={}
        Additional keyword arguments passed to `ax.quiver()` for 2D plots.
    width_pcts : list of float, default=[25, 50, 75]
        Percentile breakpoints for stratifying arrow widths by signal strength.
    width_scale : list of float, default=[0.5, 2]
        Minimum and maximum multipliers applied to the base arrow width across
        the percentile strata defined by `width_pcts`.
    cid_list : ndarray, optional
        Array of cell indices to include. Only sender-receiver pairs where both
        cells are in `cid_list` are plotted. Default: all cells.
    scaling : bool, default=True
        If True and `sem` is provided, transforms internal simulation coordinates to
        original spatial coordinates via `xc * scale + deltax`.
    ax : Axes, optional
        Target matplotlib axes object for 2D plots. Creates new figure if None.
    plotter : pyvista.Plotter, optional
        PyVista plotter for 3D visualization. Creates new plotter if None.

    Returns
    -------
    ax : Axes
        For 2D data (``dim == 2``).
    plotter : pyvista.Plotter
        For 3D data (``dim == 3``).
    """
    if sem is None:
        assert(adata is not None)
        nc = adata.shape[0]
        if signal:
            sig_mat = adata.obsp[signal]
        elif sig_mat is None:
            raise RuntimeError('signal or sig_mat are not provided')
        
        spatial_coor = adata.obsm[spatial_key]
    else:
        nc = sem.nc
        if signal:
            sig_mat = sem.adata.obsp[signal]
        elif sig_mat is None:
            raise RuntimeError('signal or sig_mat are not provided')
        
        if scaling:
            spatial_coor = sem.xc*sem.scale+sem.deltax
        else:
            spatial_coor = sem.xc
    dim = spatial_coor.shape[1]
    cid_list = _get_cid_list(cid_list, nc)

    start_points = []
    directions = []
    values = []
    indices = sig_mat.indices
    indptr = sig_mat.indptr
    data = sig_mat.data
    if callable(th):
        th = th(data)
        print('sig_mat data th',th)
    for i in cid_list:  # receiver
        for ptr in range(indptr[i],indptr[i+1]): 
            j = indices[ptr] # sender
            v = data[ptr]
            if j in cid_list and v>th: 
                start = spatial_coor[j] # sender
                end = spatial_coor[i] # receiver
                direction = (end - start)*quiver_length
                start_points.append(start)
                directions.append(direction)
                values.append(v)
    start_points = np.array(start_points)
    directions = np.array(directions)
    values = np.array(values)

    # color-strength
    # cmap = colormaps['binary']
    # norm = Normalize(vmin = np.percentile(data,5), vmax = np.percentile(data,95))
    # colors = cmap(norm(values))

    # alpha-strength
    # colors = np.zeros((values.shape[0],4))
    # v5 = np.percentile(data,5)
    # v95 = np.percentile(data,95)
    # colors[:,3] = np.clip((values-v5)/(v95-v5),0,1)

    # colors[:,3] =np.minimum( minmax_scale(values,(np.percentile(data,5), np.percentile(data,95))),1)

    # if start_points.shape[0]>0:
    
    if dim == 2:
        fig, ax = _get_axes(ax)
        if start_points.shape[0]>0:
            quiver2d_param_default = {
                'color':'k',
                # 'color':colors,
                'angles':'xy',
                'scale_units':'xy',
                'scale':1,
                'width':0.002,
                'headwidth':4,
                'headlength':5,
                'headaxislength':4.5
            }
            quiver2d_param_default.update(quiver2d_param)
            # ax.quiver(
            #     start_points[:, 0], start_points[:, 1],
            #     directions[:, 0], directions[:, 1],
            #     **quiver2d_param_default
            # )
            
            w = quiver2d_param_default['width']*np.linspace(width_scale[0],width_scale[1],width_bin_n)
            width_bins = np.linspace(values.min(), np.percentile(values,95), width_bin_n+1)
            width_bins[0] = values.min()*0.9
            # pct_values = np.percentile(values, width_pcts)
            for k in range(width_bin_n):
                if k < width_bin_n - 1:
                    I = (values >= width_bins[k]) & (values < width_bins[k + 1])
                else:
                    I = values >= width_bins[k]
                quiver2d_param_default['width'] = w[k]
                # quiver2d_param_default['color'] = colors[I,:]
                ax.quiver(
                    start_points[I, 0], start_points[I, 1],
                    directions[I, 0], directions[I, 1],
                    **quiver2d_param_default
                )
        else:
            print('zero vectors')
        return ax
    else:
        if plotter is None:
            plotter = pv.Plotter(notebook=True, window_size=[1000, 1000])
        if start_points.shape[0]>0:
            points = pv.PolyData(start_points)
            points['vectors'] = directions
            arrows = points.glyph(
                orient='vectors',
                factor=1.0,
                geom=pv.Arrow(tip_length=0.25, tip_radius=0.1, shaft_radius=0.05)
            )
            plotter.add_mesh(arrows, color='black', lighting=False,opacity=1)
        else:
            print('zero vectors')
        return plotter

def signal_direction_plot(
    sem: SEM2 = None,
    adata: Optional[AnnData] = None,
    spatial_key: Optional[str] = 'spatial',
    sig_mat: Optional[Union[csr_matrix, NDArray]] = None,
    signal: Optional[str] = None,
    th: float = 0.,
    quiver_length: float = 0.45,
    min_quiver_length: float = -1.,
    quiver2d_param: dict = {},
    width_bin_n = 3,
    width_scale = [0.5,2],
    cid_list: Union[NDArray[np.int_], NDArray[np.bool_], Sequence[int], None] = None,
    scaling: bool = True,
    ax: Optional[Axes] = None,
    rotation: Optional[Dict] = None,
):
    """
    Plot spatial directions of cell-cell communication using quiver arrows.

    For each sender-receiver pair with signal strength above `th`, draws an arrow
    centered at the midpoint of the sender-receiver segment with length
    ``(end - start) * quiver_length``. When communication is bidirectional
    (both ``sig_mat[i, j] > th`` and ``sig_mat[j, i] > th``), two opposite quiver
    arrows are overlaid at the midpoint to form a visual double-headed arrow.
    Arrow width is stratified by signal strength percentiles.

    Parameters
    ----------
    sem : SEM, SEM1, SEM2, optional
        Subcellular element method object. If provided, cell coordinates and signal
        matrix are retrieved from `sem`. Takes precedence over `adata`.
    adata : AnnData, optional
        AnnData object used when `sem` is None. Must contain spatial coordinates in
        `adata.obsm[spatial_key]` and the signal matrix in `adata.obsp[signal]`.
    spatial_key : str, default='spatial'
        Key in `adata.obsm` for spatial coordinates. Only used when `sem` is None.
    sig_mat : csr_matrix or ndarray, optional
        Signal matrix of shape `(nc, nc)`. `sig_mat[i, j]` is the signal strength
        from sender `j` to receiver `i`. Required if `signal` is not provided.
    signal : str, optional
        Key in `sem.adata.obsp` or `adata.obsp` to retrieve the signal matrix.
        Takes precedence over `sig_mat`.
    th : float or callable, default=0.
        Signal strength threshold. Sender-receiver pairs with signal <= `th` are not
        plotted. If callable, it is applied to all signal data values to compute `th`.
    quiver_length : float, default=0.45
        Fraction of the sender-to-receiver vector used as arrow length.
        1.0 means the arrow tip reaches the receiver centroid.
    quiver2d_param : dict, default={}
        Additional keyword arguments passed to `ax.quiver()`
    width_pcts : list of float, default=[25, 50, 75]
        Percentile breakpoints for stratifying arrow widths by signal strength.
    width_scale : list of float, default=[0.5, 2]
        Minimum and maximum multipliers applied to the base arrow width across
        the percentile strata defined by `width_pcts`.
    cid_list : ndarray, optional
        Array of cell indices to include. Only sender-receiver pairs where both
        cells are in `cid_list` are plotted. Default: all cells.
    scaling : bool, default=True
        If True and `sem` is provided, transforms internal simulation coordinates to
        original spatial coordinates via `xc * scale + deltax`.
    ax : Axes, optional
        Target matplotlib axes object. Creates new figure if None.
        
    Returns
    -------
    ax : Axes
    """
    if sem is None:
        assert(adata is not None)
        nc = adata.shape[0]
        if signal:
            sig_mat = adata.obsp[signal].copy()
        elif sig_mat is None:
            raise RuntimeError('signal or sig_mat are not provided')
        
        spatial_coor = adata.obsm[spatial_key].copy()
    else:
        nc = sem.nc
        if signal:
            sig_mat = sem.adata.obsp[signal].copy()
        elif sig_mat is None:
            raise RuntimeError('signal or sig_mat are not provided')
        
        if scaling:
            spatial_coor = sem.xc*sem.scale+sem.deltax
        else:
            spatial_coor = sem.xc.copy()
    cid_list = _get_cid_list(cid_list, nc)
    cid_set = set(cid_list.tolist())

    if rotation is not None:
        center = rotation['center']
        theta = rotation['theta']
        RT = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).T
        spatial_coor = (spatial_coor-center) @ RT + center

    # Unidirectional: one quiver centered at midpoint
    uni_starts = []
    uni_dirs = []
    uni_values = []
    # Bidirectional: two opposite quivers overlapping at midpoint
    bi_starts_fwd = []   # midpoint - direction/2, direction = fwd (j->i)
    bi_dirs_fwd = []
    bi_starts_rev = []   # midpoint + direction/2, direction = -fwd (i->j)
    bi_dirs_rev = []
    bi_values_fwd = []
    bi_values_rev = []

    indices = sig_mat.indices
    indptr = sig_mat.indptr
    data = sig_mat.data
    if callable(th):
        th = th(data)
        print('sig_mat data th', th)

    processed_pairs = set()
    a = 0.5
    for i in cid_list:  # receiver
        for ptr in range(indptr[i], indptr[i+1]):
            j = indices[ptr]  # sender
            v_ji = data[ptr]  # signal j -> i
            if j not in cid_set or v_ji < th:
                continue
            pair = (min(i, j), max(i, j))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            start = spatial_coor[j]   # sender centroid
            end   = spatial_coor[i]   # receiver centroid
            midpoint  = (start + end) * 0.5
            direction = (end - start) * quiver_length  # j -> i
            l = np.linalg.norm(direction)
            

            v_ij = sig_mat[j, i]  # signal i -> j
            if v_ij > th:
                # Bidirectional: two opposite quivers centered at midpoint
                if l<min_quiver_length:
                    direction*= (min_quiver_length/l)
                bi_starts_fwd.append(midpoint)
                bi_dirs_fwd.append(direction*a)
                bi_values_fwd.append(v_ji)
                bi_starts_rev.append(midpoint)
                bi_dirs_rev.append(-direction*a)
                bi_values_rev.append(v_ij)
            else:
                # Unidirectional j -> i: quiver start = midpoint - direction/2
                # if l<(min_quiver_length):
                #     uni_starts.append(midpoint - direction*min_quiver_length/l* 0.55)
                #     uni_dirs.append(direction*min_quiver_length/l)
                # else:
                uni_starts.append(midpoint - direction * 0.4)
                uni_dirs.append(direction)
                uni_values.append(v_ji)

    uni_starts = np.array(uni_starts) if uni_starts else np.empty((0, 2))
    uni_dirs   = np.array(uni_dirs)   if uni_dirs   else np.empty((0, 2))
    uni_values = np.array(uni_values) if uni_values else np.empty(0)
    bi_starts_fwd = np.array(bi_starts_fwd) if bi_starts_fwd else np.empty((0, 2))
    bi_dirs_fwd   = np.array(bi_dirs_fwd)   if bi_dirs_fwd   else np.empty((0, 2))
    bi_starts_rev = np.array(bi_starts_rev) if bi_starts_rev else np.empty((0, 2))
    bi_dirs_rev   = np.array(bi_dirs_rev)   if bi_dirs_rev   else np.empty((0, 2))
    bi_values_fwd     = np.array(bi_values_fwd)     if bi_values_fwd     else np.empty(0)
    bi_values_rev     = np.array(bi_values_rev)     if bi_values_rev     else np.empty(0)

    fig, ax = _get_axes(ax)

    # all_values = uni_values
    all_values = np.concatenate([uni_values, bi_values_fwd, bi_values_rev])
    if all_values.shape[0] == 0:
        print('zero vectors')
        return ax

    quiver2d_param_default = {
        'color': 'k',
        'angles': 'xy',
        'scale_units': 'xy',
        'scale': 1,
        'width': 0.002,
        'headwidth': 4,
        'headlength': 5,
        'headaxislength': 4.5
    }
    quiver2d_param_default.update(quiver2d_param)

    base_width = quiver2d_param_default['width']
    w = base_width * np.linspace(width_scale[0], width_scale[1], width_bin_n)
    width_bins = np.linspace(all_values.min(), np.percentile(all_values,95), width_bin_n+1)
    # width_bins[0] = all_values.min()*0.9

    def _draw_quiver(starts, dirs, values):
        for k in range(width_bin_n):
            if k < width_bin_n - 1:
                I = (values >= width_bins[k]) & (values < width_bins[k + 1])
            else:
                I = values >= width_bins[k]
            quiver2d_param_default['width'] = w[k]
            ax.quiver(
                starts[I, 0], starts[I, 1],
                dirs[I, 0],   dirs[I, 1],
                **quiver2d_param_default
            )

    if uni_starts.shape[0] > 0:
        _draw_quiver(uni_starts, uni_dirs, uni_values)

    if bi_starts_fwd.shape[0] > 0:
        _draw_quiver(bi_starts_fwd, bi_dirs_fwd, bi_values_fwd)
        _draw_quiver(bi_starts_rev, bi_dirs_rev, bi_values_rev)

    return ax

def vis_contact_signal(
    sem: Optional[SEM2] = None,
    adata: Optional[AnnData] = None,
    spatial_key: Optional[str] = 'spatial',
    sig_mat: Optional[Union[csr_matrix, NDArray]] = None,
    signal: Optional[str] = None,
    cid_list: Union[NDArray[np.int_], NDArray[np.bool_], Sequence[int], None] = None,
    scaling: bool = True,
    line_width: Union[float, Iterable[float]] = 1,
    line_color: ColorType = 'k',
    line_alpha: Union[float, Iterable[float]] = 1,
    ax: Optional[Axes] = None
) -> Axes:
    """
    Visualize contact signals or relationships between cells

    Parameters
    ----------
    sem : SEM
        A subcellular element method object.
    sig_mat : csr_matrix or ndarray, optional
        Signal matrix to visualize. If `signal` is provided, this parameter will be ignored, 
        and the signal matrix will be retrieved from `sem.adata.obsp[signal]`.
        
        If `sig_mat` and `signal` both are None, the contact matrix `sem.contact_matrix` will be visualized.
    signal : str, optional
        Key for signal matrix in `sem.adata.obsp`. If given, `sig_mat` will be ignored.
    cid_list : ndarray, optional
        Array of index for cells to be visualized. Default: all cells
    scaling : bool, default=True
        Scale coordinates back to original data if True, otherwise visualize directly
    line_width : float, default=1
        Cell-cell contacts line width
    line_color : color, default='k'
        Cell-cell contacts line color
    line_alpha : float, default=1
        Cell-cell contacts line opacity, 0 (fully transparent), 1 (fully opaque)
    ax : Axes, optional
        Target matplotlib axes object. Creates new figure if None

    Returns
    ----------
    ax : Axes
    """
    fig, ax = _get_axes(ax)
    if sem is None:
        assert(adata is not None)
        nc = adata.shape[0]
        if signal:
            sig_mat = adata.obsp[signal]
        spatial_coor = adata.obsm[spatial_key]
    else:
        nc = sem.nc
        if signal:
            sig_mat = sem.adata.obsp[signal]
        elif sig_mat is None:
            if sem.contact_matrix is None:
                print('compute cell-cell contact')
                sem.compute_contact()
            sig_mat = sem.contact_matrix

        if scaling:
            spatial_coor = sem.xc*sem.scale+sem.deltax
        else:
            spatial_coor = sem.xc
    cid_list = _get_cid_list(cid_list, nc)
    seg = []
    if isinstance(line_width, (list, tuple)):
        data = sig_mat.data
        linewidths = np.abs(data)/data.max()*line_width[1]
    else:
        linewidths = line_width
    if isinstance(line_alpha, (list, tuple)):
        data = sig_mat.data
        linealphas = np.abs(data)/data.max()*line_alpha[1]
    else:
        linealphas = line_alpha
    if isspmatrix(sig_mat):
        indices = sig_mat.indices
        indptr = sig_mat.indptr
        for i in cid_list:
            for j in indices[indptr[i]:indptr[i+1]]:
                if j in cid_list:
                    seg.append([spatial_coor[i],spatial_coor[j]])
    else:
        for j in cid_list:
            sender_i = np.where(sig_mat[:,j]>0)[0]
            for i in sender_i:
                if i in cid_list:
                    seg.append([spatial_coor[i],spatial_coor[j]])
    lc = LineCollection(seg, linewidths=linewidths, colors=line_color, alpha=linealphas)
    ax.add_collection(lc)
    return ax

def cluster_comm_plot(
    adata: AnnData,
    cluster_key:str,
    signal_key:str,
    prefix:str = 'mechano_',
    ms = 100, 
    pvalue:float = 0.05,
    cb_label: Optional[str] = '',
    save_name: Optional[str] = None,
    rotation = 90,
    show: bool=True
) -> Axes:
    
    cluster_signal_key = f'{cluster_key}_{prefix}{signal_key}'
    mean_df = adata.uns[cluster_signal_key]['communication_matrix']
    p_df = adata.uns[cluster_signal_key]['communication_pvalue']
    senders = mean_df.index.tolist()
    receivers = mean_df.columns.tolist()
    norm = Normalize(vmin=mean_df.min().min(),vmax=mean_df.max().max()*1.1)
    fig,ax = plt.subplots(figsize=(8,3))
    orig = plt.get_cmap('Reds')
    new_colors = orig(np.linspace(0, 0.9, 256))
    cmap = LinearSegmentedColormap.from_list(name = 'truncate_Reds',colors = new_colors)
    for i, s in enumerate(senders):
        for j, r in enumerate(receivers):
            if mean_df.loc[s, r] > 0:
                ax.scatter(j, i, marker  = 's', s = ms, c = mean_df.loc[s, r], cmap = cmap, norm = norm, edgecolors = "k", linewidths = 0.5)
            if p_df.loc[s, r] < pvalue:
                ax.scatter(j, i,marker = '*',color='k', s = 20,linewidths=0.2)

    ax.set_xticks(range(len(receivers)))
    ax.set_xticklabels(receivers, rotation=rotation, horizontalalignment = 'right',rotation_mode='anchor',va='center')
    ax.tick_params(axis='x', which='major', pad=1) 
    ax.set_yticks(range(len(senders)))  
    ax.set_yticklabels(senders)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect('equal')
    ax.set_xlim(-0.6, len(receivers) - 0.5)
    ax.set_ylim(len(senders) - 0.4, -0.5)
    ax.grid(True, linestyle=':', linewidth=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(sm,cax=cax, ticks=[], label=cb_label)
    cb.outline.set_visible(False)
    _save_close(fig,save_name,show)
    return ax

def sptial_domain_comm_plot(
    sem: SEM2,
    df_pvalue: pd.DataFrame,
    cluster_key: str,
    ax: Optional[Axes] = None,
    threshold: float = 0.05,
    loop_angle_default: float = 0,
    node_size: float = 1.8,
    loop_size: float = 3.5,
    arror_size: float = 25,
    arrowstyle = ArrowStyle('-|>',head_width=0.2),
    color = np.zeros(3),
    edge_width: float = 3.2,
    node_line_width: float = 1,
) -> Axes:

    fig, ax = _get_axes(ax)
    xc = sem.xc*sem.scale+sem.deltax
    nc = xc.shape[0]
    cmap = dict(zip(sem.adata.obs[cluster_key].cat.categories,sem.adata.uns[f'{cluster_key}_colors']))
    # w = sig_mat.sum(axis=0).A1
    w = np.ones(nc)
    x_cluster_map = dict()
    cmap1 = dict()
    for ct1 in sem.adata.obs[f'{cluster_key}_mc'].cat.categories:
        I = sem.adata.obs[f'{cluster_key}_mc'] == ct1
        x_cluster = np.average(xc[I,:],weights=w[I],axis=0)
        ct = sem.adata.obs[cluster_key][I].unique()[0]
        if I.sum()>=5:
            cmap1[ct1] = cmap[ct]
            d = np.linalg.norm(x_cluster - xc[I,:],2,axis=1)
            idx = np.argmin(d)
            xc_temp = xc[I,:]
            x_cluster_map[ct1] = xc_temp[idx,:]
    cell_types = list(x_cluster_map.keys())

    # node_radius = np.sqrt(node_size / np.pi)  # 转换为半径
    node_radius = node_size/4
    loop_radius = np.sqrt(loop_size / np.pi)  # 转换为半径
    for cell_type in cell_types:
        x, y = x_cluster_map[cell_type]
        rect = Rectangle((x-node_size/2, y-node_size/2), node_size, node_size, linewidth=node_line_width, edgecolor='k', facecolor=cmap1[cell_type],zorder=2)
        ax.add_patch(rect)
        # circle = Circle(
        #     (x, y), node_radius, 
        #     facecolor=cmap1[cell_type],
        #     edgecolor='k',
        #     linewidth=1,
        #     zorder=2
        #     )
        # ax.add_patch(circle)

    angle_edge = defaultdict(list)
    for target in df_pvalue.index:
        for source in df_pvalue.columns:
            if df_pvalue.loc[target, source] < threshold and source != target:
                x_src, y_src = x_cluster_map[source]
                x_tgt, y_tgt = x_cluster_map[target]
                dx = x_tgt - x_src
                dy = y_tgt - y_src
                dist = np.sqrt(dx**2 + dy**2)
                
                # if dist > 0:
                ux = dx / (dist+1e-10)
                uy = dy / (dist+1e-10)
                angle_edge[source].append([ux,uy])
                angle_edge[target].append([-ux,-uy])
                # 起点和终点（考虑节点半径）
                start_x = x_src + node_radius * ux
                start_y = y_src + node_radius * uy
                end_x = x_tgt - node_radius * ux
                end_y = y_tgt - node_radius * uy
                
                s = 5 if df_pvalue.loc[source, target] < threshold else 2
                    
                arrow = FancyArrowPatch(
                    (start_x, start_y),
                    (end_x, end_y),
                    arrowstyle=arrowstyle,
                    mutation_scale=arror_size,
                    linewidth=0.1,
                    color=color,
                    zorder=2,
                    shrinkA=s,
                    shrinkB=2,
                    alpha = 1,
                )
                ax.add_patch(arrow)

                line = FancyArrowPatch(
                    (start_x, start_y),
                    (end_x, end_y),
                    arrowstyle='-',
                    mutation_scale=arror_size,
                    linewidth=edge_width,
                    color=color,
                    zorder=2,
                    shrinkA=s,
                    shrinkB=s+1,
                    alpha = 1,
                )
                ax.add_patch(line)

    for source in df_pvalue.index:
        if df_pvalue.loc[source, source] < threshold:

            u = np.array(angle_edge[source])
            if u.shape[0]==0:
                # loop_angle = 0 # 自环位置角度（度）
                loop_angle = loop_angle_default
                ux = np.cos(loop_angle)
                uy = np.sin(loop_angle)
            else:
                u = -np.mean(np.array(angle_edge[source]),axis=0)
                ux, uy = u / np.linalg.norm(u)
                loop_angle = np.arctan2(uy, ux)/np.pi * 180

            x_src, y_src = x_cluster_map[source]
            # loop_radius = node_radius * 1.4 # 自环半径
            
            # 计算自环中心位置（在节点外侧）
            # angle_rad = np.radians(180)
            angle_rad = np.radians(loop_angle)
            loop_center_x = x_src + (node_radius + loop_radius/2) * ux#np.cos(angle_rad) 
            loop_center_y = y_src + (node_radius + loop_radius/2) * uy#np.sin(angle_rad)
            
            arc = Arc(
                (loop_center_x, loop_center_y), 
                loop_radius * 2, loop_radius * 2,
                angle=loop_angle+180, theta1=30, theta2=330,
                linewidth=edge_width, color=color, zorder=1,
                alpha = 1
                )
            ax.add_patch(arc)

            # arrow_angle = np.radians(loop_angle)
            arrow_x = loop_center_x + loop_radius * np.cos(angle_rad)
            arrow_y = loop_center_y + loop_radius * np.sin(angle_rad)

            dx = -loop_radius * np.sin(angle_rad) * 0.03 * arror_size
            dy = loop_radius * np.cos(angle_rad) * 0.03 * arror_size

            arrow = FancyArrowPatch(
                (arrow_x, arrow_y),
                (arrow_x+dx, arrow_y+dy),
                arrowstyle=arrowstyle, mutation_scale=arror_size,
                linewidth=0.1, color=color, 
                zorder=1,
                alpha = 1
            )
            ax.add_patch(arrow)
    ax.autoscale()
    return ax

def contact_plot(
    ctri: cell_tri_dev,
    vis_key: Literal['mechanical interaction', 'tension', 'tension1'],
    mean_edge_value: bool = True,
    line_width: float = 2,
    cid_list: Union[NDArray[np.int_], NDArray[np.bool_], Sequence[int], None] = None,
    pair_list: Optional[NDArray[np.int_]] = None,
    norm: Union[Normalize, None] = None,
    cmap_name: str = 'inferno',
    show_axis: bool = False,
    ax: Optional[Axes] = None,
    rotation: Optional[Dict] = None,
    return_norm: bool = False,
):
    """
    Plot cell-cell contact mechanical signals as colored line segments.

    Requires ``contact_analysis`` to be run beforehand on the ``ctri`` object.

    Parameters
    ----------
    ctri : cell_tri or cell_tri_dev
        Cell triangulation object containing contact geometry and force data
    vis_key : {'mechanical interaction', 'tension', 'tension1'}
        Signal to visualize:

        - ``'mechanical interaction'`` — cell-cell mechanical interaction force
          (also renders junction segments separately)
        - ``'tension'`` — contact edge tension
        - ``'tension1'`` — mean tension per contact pair
    mean_edge_value : bool, default=True
        If True, assign the mean signal value uniformly to all segments of each
        contact edge; if False, use per-segment values
    line_width : float, default=2
        Width of the contact line segments
    cid_list : array-like, optional
        Cell indices to include. Default: all cells
    pair_list : ndarray of shape (n, 2), optional
        Restrict display to specific cell-pair indices. Default: all pairs
    norm : Normalize, optional
        Matplotlib normalization object for colormap mapping.
        Default: 5th–95th percentile of the signal
    cmap_name : str, default='inferno'
        Valid matplotlib colormap name
    show_axis : bool, default=False
        Show axis
    ax : Axes, optional
        Target matplotlib axes object. Creates new figure if None.
    rotation : dict, optional
        Rotate coordinates before plotting. Expected keys:

        - ``'center'`` — rotation center, array of shape (2,)
        - ``'theta'`` — rotation angle in radians
    return_norm : bool, default=False
        If True, also return the normalization object and colormap

    Returns
    ----------
    ax : Axes
        If ``return_norm=False``
    (ax, norm, cmap) : tuple
        If ``return_norm=True``
    """
    fig, ax = _get_axes(ax)
    cid_list = _get_cid_list(cid_list, ctri.nc)

    cmap = colormaps[cmap_name]

    if rotation is not None:
        center = rotation['center']
        theta = rotation['theta']
        RT = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).T
        enable_rotation = True
    else:
        enable_rotation = False
    
    assert vis_key in ('mechanical interaction', 'tension', 'tension1')
    junction_v_dict = None
    if vis_key == 'mechanical interaction':
        contact_v_dict = ctri.contact_f_dict.copy()
        for ij in contact_v_dict:
            contact_v_dict[ij] = np.mean(np.array(contact_v_dict[ij]),axis=1)
        junction_v_dict = ctri.junction_f_dict.copy()
    elif vis_key == 'tension':
        contact_v_dict = ctri.contact_ten_dict.copy()
    elif vis_key == "tension1":
        contact_v_dict = ctri.contact_ten_dict1.copy()
        for ij in contact_v_dict:
            contact_v_dict[ij] = np.mean(np.array(contact_v_dict[ij]),axis=1)


    contact_midedge = []
    contact_ten_m = []
    for ij in ctri.contact_midedge_dict:
        if pair_list is not None:
            if ~(np.array(ij) == pair_list).all(axis=1).any():
                continue
        if np.isin(np.array(ij) , cid_list).all():
        # if (np.array(ij) == pair_list).all(axis=1).any():
            contact_midedge.append(ctri.contact_midedge_dict[ij])
            if mean_edge_value:
                len_edge = len(ctri.contact_midedge_dict[ij])
                contact_ten_m.append(np.mean(contact_v_dict[ij]).repeat(len_edge))
            else:
                contact_ten_m.append(contact_v_dict[ij])
    contact_midedge = np.concatenate(contact_midedge)
    contact_ten_m = np.concatenate(contact_ten_m)
    if enable_rotation:
        contact_midedge = (contact_midedge-center) @ RT + center

    if norm is None:
        norm = Normalize(vmin=np.percentile(contact_ten_m,5), vmax=np.percentile(contact_ten_m,95))

    lc = LineCollection(
        contact_midedge,
        cmap=cmap,
        norm=norm,
        linewidths=line_width, 
        antialiased=True,
        path_effects=[path_effects.Stroke(capstyle="round",joinstyle="round")],
    )
    lc.set_array(contact_ten_m)
    ax.add_collection(lc)

    if junction_v_dict is not None:
        contact_midedge = []
        contact_ten_m = []
        for ij in ctri.junction_midedge_dict:
            if np.isin (np.array(ij) , cid_list).all():
                contact_midedge.append(ctri.junction_midedge_dict[ij])
                contact_ten_m.append(junction_v_dict[ij])
        contact_midedge = np.concatenate(contact_midedge)
        contact_ten_m = np.concatenate(contact_ten_m)
        if enable_rotation:
            contact_midedge = (contact_midedge-center) @ RT + center
        lc = LineCollection(
            contact_midedge,
            cmap=cmap,
            norm=norm,
            linewidths=line_width, 
            antialiased=True,
            path_effects=[path_effects.Stroke(capstyle="round",joinstyle="round")],
        )
        lc.set_array(contact_ten_m)
        ax.add_collection(lc)
    
    _set_axes(ax,show_axis)
    if return_norm:
        return ax, norm, cmap
    else:
        return ax

def highlight_cell(
    sem: SEM2, 
    cid_list,
    color: ColorType,
    ax: Optional[Axes] = None, 
    ns:int = 8,
    r: float = 10.,
    ar: Optional[float] = None, #80.
    face_alpha: float = 0.1,
    edge_alpha: float = 0.9,
    lw: float = 1.5,
    zorder: int=2,
    rotation: Optional[Dict] = None,
    scaling: bool = True,
    show_axis: bool = True,
):
    """
    Highlight a group of cells with a unified color overlay using alpha shape.

    Parameters
    ----------
    sem : SEM
        Subcellular element method object
    cid_list : array-like
        Cell indices to highlight. Accepts int, bool array, or list of indices.
        If None, all cells are selected.
    color : color
        Highlight color. Accepts:

        - Named color string, e.g. ``'r'``, ``'blue'``
        - Hex string, e.g. ``'#FF5733'``
        - RGB array/list/tuple with values in [0, 1], e.g. ``[0.2, 0.6, 0.8]``
    ax : Axes, optional
        Target matplotlib axes object. Creates new figure if None.
    ns : int, default=8
        Number of elements per segment used by ``AlphaShape``
    r : float, default=10.0
        Neighbor search radius for ``AlphaShape``
    ar : float, default=80.0
        Alpha radius controlling the concavity of the shape boundary
    face_alpha : float, default=0.1
        Face opacity of the highlighted region, 0 (fully transparent), 1 (fully opaque)
    edge_alpha : float, default=0.9
        Edge opacity of the highlighted region, 0 (fully transparent), 1 (fully opaque)
    lw : float, default=1.5
        Boundary line width
    scaling : bool, default=True
        If True, transform element coordinates from simulation space to spatial units
    show_axis : bool, default=True
        Show axis

    Returns
    ----------
    ax : Axes
    """

    fig, ax = _get_axes(ax)
    cid_list = _get_cid_list(cid_list, sem.nc)
    xe = _get_xe(sem,cid_list,scaling)
    if ar is None:
        ar = sem.alphashape[cid_list[0]].alpha
        for i in cid_list[1:]:
            ar = max(ar, sem.alphashape[i].alpha)
        # ar*=1.5

    if rotation:
        center = rotation['center']
        theta = rotation['theta']
        RT = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).T
        xe = (xe-center) @ RT + center

    shp = AlphaShape(xe,ns=ns,r=r)
    shp.update(ar)
    # shp.close_hole()
    shp_boundary = shp.get_boundarys()
    
    color = np.array(to_rgb(color))
    facecolor = tuple(np.insert(color, 3, face_alpha))
    edgecolor = tuple(np.insert(color, 3, edge_alpha))
    path = _boundaries_to_path(shp_boundary)
    patch = PathPatch(
        path,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    _set_axes(ax,show_axis=show_axis)
    return ax

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

def plot_grid_signal_direction(
    Xgrid, Vgrid,
    rs = Literal['receiver','sender'],
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

def plot_grid_signal_tensor(
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

def draw_colorbar(
    cmap_name: str = 'Reds', 
    cmap = None,
    norm: Optional[Normalize] = None,
    vmax: float = 0.,
    vmin: float = 1.,
    orientation: Literal['horizontal','vertical'] = 'horizontal',
    label: Optional[str] = None,
    label_size: float = 15,
    ticks: Optional[List] = None,
    ticklabels: Optional[List] = None,
    ticks_size: float = 15,
    save_name: Optional[str] = None,
    ax: Optional[Axes] = None,
):
    if norm is None:
        norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
    cmap = colormaps[cmap_name] if cmap is None else cmap
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    figsize=(3, 0.3) if orientation == 'horizontal' else (0.3, 3)
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    else:
        fig=ax.figure

    cb = plt.colorbar(sm, cax=ax, orientation=orientation)# 'vertical'
    cb.ax.tick_params(labelsize=ticks_size)

    if ticks is not None:
        cb.set_ticks(ticks)

    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
        
    if label is not None:
        cb.set_label(label,fontsize=label_size)

    if save_name is not None:
        fig.savefig(save_name, dpi=500, bbox_inches='tight', transparent=True)

    return ax

def draw_legend(
    adata: Optional[AnnData] = None,
    cluster_key: Optional[str] = None,
    cmap: Optional[dict] = None,
    ncol: int = 2,
    loc: str = 'center left',
    fontsize: float = 14.,
    save_name: Optional[str] = None,
    cid_list: Optional[NDArray[np.int_]] = None,
    rename_map: Optional[Dict] = None,
    ax: Optional[Axes] = None,
):
    fig, ax = _get_axes(ax)
    legend_patches = []
    if adata is None:
        assert cmap is not None
        for ct in cmap:
            # if ct in cat_list: # 保证原本顺序
            legend_patches.append(Patch(color=cmap[ct],label=ct))
    else:
        cid_list = _get_cid_list(cid_list, adata.shape[0])
        color_list_all = adata.uns[f'{cluster_key}_colors']
        if rename_map is None:
            ct_df = adata.obs[cluster_key]
        else:
            ct_df = adata.obs[cluster_key].cat.rename_categories(
                lambda x: rename_map.get(x, x)
            )
        cat_list_all = ct_df.cat.categories
        cat_list = adata.obs[cluster_key][cid_list].unique()
        for i,ct in enumerate(cat_list_all):
            if ct in cat_list: # 保证原本顺序
                legend_patches.append(Patch(color=color_list_all[i],label=ct))
    
    ax.set_axis_off()
    ax.legend(
        handles=legend_patches,
        loc=loc,
        ncol=ncol,
        frameon=False,
        fontsize = fontsize,
    )
    plt.tight_layout()
    if save_name is not None:
        fig.savefig(save_name, dpi=500, bbox_inches='tight', transparent=True)
    return ax


def _boundaries_to_path(shp_boundary) -> MplPath:
    """
    Convert a list of boundary coordinate arrays into a compound matplotlib Path.

    Outer boundaries are oriented CCW; hole boundaries (whose centroid lies inside
    another boundary) are oriented CW.  The resulting compound path uses the
    nonzero winding rule so that holes are correctly cut out when rendered.
    """
    boundaries = [np.asarray(b) for b in shp_boundary]
    n = len(boundaries)

    # Classify each boundary as hole or outer
    is_hole = [False] * n
    for i in range(n):
        centroid_i = boundaries[i].mean(axis=0)
        for j in range(n):
            if i != j and MplPath(boundaries[j]).contains_point(centroid_i):
                is_hole[i] = True
                break

    all_verts = []
    all_codes = []
    for boundary, hole in zip(boundaries, is_hole):
        pts = boundary.copy()
        # Shoelace signed area: positive → CCW, negative → CW
        x, y = pts[:, 0], pts[:, 1]
        sa = 0.5 * (np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]) + x[-1]*y[0] - x[0]*y[-1])
        # Outer must be CCW; hole must be CW
        if (hole and sa > 0) or (not hole and sa < 0):
            pts = pts[::-1]
        closed = np.vstack([pts, pts[0]])
        m = len(closed)
        codes = np.full(m, MplPath.LINETO, dtype=np.uint8)
        codes[0] = MplPath.MOVETO
        codes[-1] = MplPath.CLOSEPOLY
        all_verts.append(closed)
        all_codes.append(codes)

    return MplPath(np.vstack(all_verts), np.concatenate(all_codes))


def _get_axes(
    ax: Optional[Axes] = None,
    dim = 2
) -> Tuple[Figure, Axes]:
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

def _get_cid_list(
    cid_list: Union[NDArray[np.int_], NDArray[np.bool_], Sequence[int], None],
    nc: int,
) -> NDArray:
    if cid_list is None:
        cid_list = np.arange(nc)
    elif isinstance(cid_list, np.ndarray):
        if cid_list.dtype == np.bool_:
            cid_list = np.where(cid_list)[0]
    else:
        cid_list = np.array(cid_list)
    return cid_list

def _get_xe(
    sem: Union[SEM2, SEM3], 
    cid_list: Union[NDArray, None],
    scaling: bool,
) -> NDArray:
    if cid_list is None:
        xe = sem.xe*sem.scale+sem.deltax if scaling else sem.xe
    else:
        xe = []
        for cid in cid_list:
            xe.append(sem.get_elements(cid, scaling))
        xe = np.vstack(xe)
    return xe

def _add_colorbar(fig, ax, cmap, norm):
    """add colorbar"""
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.4) #to do fix colorbar width 
    cb = plt.colorbar(sm,cax=cax)
    cb.ax.tick_params(labelsize=20)

def _set_axes(ax, show_axis):
    """aspect equal, axis off, invert yaxis"""
    ax.set_aspect('equal', adjustable='box')
    ax.autoscale(tight=True)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    if not show_axis:
        ax.set_axis_off()

def _get_arr1(
        sem: Union[SEM2, SEM3],
        vis_key: Union[str,None],
        arr: Union[NDArray, pd.Series, None],
        summary: str,
        obsm_key: Optional[str]
):
    
    if (sem.adata is not None) & (vis_key is not None):

        if summary == 'gene' and vis_key in sem.adata.var_names:
            arr = sem.adata[:,vis_key].X.toarray()[:,0] # retrieve gene expression

        elif summary == 'cell' and obsm_key in sem.adata.obsm:
            arr = sem.adata.obsm[obsm_key][vis_key].to_numpy() # retrieve signal

        elif summary in sem.adata.obsm:
            if vis_key in sem.adata.obsm[summary]:
                arr = sem.adata.obsm[summary][vis_key].to_numpy() # retrieve signal

        if arr is None and vis_key in sem.adata.obs:
            arr = sem.adata.obs[vis_key] # retrieve adata.obs
    return arr

def _get_arr(
    sem: Union[SEM2, SEM3],
    vis_key: Union[str,None],
    arr: Union[NDArray, pd.Series, None],
    summary: str
):
    # used in element plot, deprecate
    if (sem.adata is not None) & (vis_key is not None):
        if summary == 'gene' and vis_key in sem.adata.var_names:
            arr = sem.adata[:,vis_key].X.toarray()[:,0] # retrieve gene expression
        elif summary == 'sender' and 'sender_signal' in sem.adata.obsm and vis_key in sem.adata.obsm['sender_signal']:
            arr = sem.adata.obsm['sender_signal'][vis_key].to_numpy() # retrieve sender signal
        elif summary == 'receiver' and 'receiver_signal' in sem.adata.obsm and vis_key in sem.adata.obsm['receiver_signal']:
            arr = sem.adata.obsm['receiver_signal'][vis_key].to_numpy() # retrieve receiver signal
        elif summary == 'receiver_mechano' and 'receiver_mechano_signal' in sem.adata.obsm and vis_key in sem.adata.obsm['receiver_mechano_signal']:
            arr = sem.adata.obsm['receiver_mechano_signal'][vis_key].to_numpy() # retrieve receiver signal
        elif vis_key in sem.adata.obs:
            arr = sem.adata.obs[vis_key] # retrieve adata.obs
    return arr

def _get_cat_arr_color(
    sem: Union[SEM2, SEM3],
    arr: pd.Series,
    cid_list: NDArray,
    vis_key: str,
    cmap_name: str,
    palette: Optional[Dict]=None
):
    
    cat_code = arr.cat.codes[cid_list]
    cat_list = arr.cat.categories.to_list()

    if palette is not None:
        color_list = [palette[x] for x in cat_list]
        color_list = np.array([to_rgb(x) for x in color_list])
    elif (vis_key+'_colors') in sem.adata.uns:
        # use cluster color in the adata
        color_list = sem.adata.uns[vis_key+'_colors']
        color_list = np.array([to_rgb(x) for x in color_list])
    else:
        cmap = colormaps[cmap_name]
        color_list = cmap( np.linspace( 0,1,len(cat_list) ) )[:,:3]
    
    # nan in arr
    if np.any(cat_code==-1):
        cat_code[cat_code==-1] = cat_code.max()+1
        cat_list.append('NA')
        color_list = np.vstack((color_list, 0.9*np.ones(3)))
    
    return cat_code, cat_list, color_list

def _save_close(fig,save_name,show):
    if save_name is not None:
        fig.savefig(save_name, dpi=500, bbox_inches='tight', transparent=True)
        if not show:
            plt.close()