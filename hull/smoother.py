import numpy as np
from scipy import spatial as sp_spatial
from hull.voxelize.voxelize import voxelize
from visualize_results import MultiView
from matplotlib import pyplot as plt
from scipy.ndimage import binary_fill_holes
from scipy.ndimage.morphology import binary_erosion


def delaunay(volume):
    coords = np.argwhere(volume == 1)

    min_z, min_y, min_x = coords[:, 0].min(), coords[:, 1].min(), coords[:, 2].min()
    max_z, max_y, max_x = coords[:, 0].max(), coords[:, 1].max(), coords[:, 2].max()

    kernel_size = 22
    stride = 18
    th = 9000

    smooth_vol = np.zeros_like(volume)

    z_start = min_z
    while z_start < max_z:
        y_start = min_y
        while y_start < max_y:
            x_start = min_x
            while x_start < max_x:

                v = coords[
                    (coords[:, 1] > y_start) & (coords[:, 1] < y_start + kernel_size) &
                    (coords[:, 0] > z_start) & (coords[:, 0] < z_start + kernel_size) &
                    (coords[:, 2] > x_start) & (coords[:, 2] < x_start + kernel_size)
                    ]

                # meshing is executed if we have at least 3 points
                if v.size < 9:
                    # if v.size > 0:
                    #     smooth_vol[v[:, 0], v[:, 1], v[:, 2]] = 1
                    x_start += stride
                    continue

                if v[:, 0].max() == v[:, 0].min() or v[:, 1].max() == v[:, 1].min() or v[:, 2].max() == v[:, 2].min():
                    x_start += stride
                    continue

                hull = sp_spatial.ConvexHull(v, incremental=True).simplices
                # mlab.triangular_mesh(v[:, 2], v[:, 1], v[:, 0], hull, color=(0, 1, 0))

                # filtering biggest tringles
                # tri = [v for v in v[hull] if abs(np.linalg.det(v))/2 < th]
                # tri = np.stack(tri)
                tri = v[hull]

                # voxellization
                if tri.size > 0:
                    for z, y, x in voxelize(tri):
                        smooth_vol[z, y, x] = 1
                x_start += stride
            y_start += stride
        z_start += stride

    return smooth_vol