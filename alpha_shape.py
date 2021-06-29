from eval import Eval
import json
from hull.voxelize.voxelize import voxelize
from scipy.spatial import Delaunay
import numpy as np
from collections import defaultdict
from scipy.ndimage import binary_fill_holes
import os
import pathlib
from glob import glob
from scipy.spatial import cKDTree


def alpha_shape_3D(pos, alpha):
    """
    Compute the alpha shape (concave hull) of a set of 3D points.
    Parameters:
        pos - np.array of shape (n,3) points.
        alpha - alpha value.
    return
        outer surface vertex indices, edge indices, and triangle indices
    """

    tetra = Delaunay(pos)
    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = np.take(pos, tetra.vertices, axis=0)
    normsq = np.sum(tetrapos ** 2, axis=2)[:, :, None]
    ones = np.ones((tetrapos.shape[0], tetrapos.shape[1], 1))
    a = np.linalg.det(np.concatenate((tetrapos, ones), axis=2))
    Dx = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [1, 2]], ones), axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 2]], ones), axis=2))
    Dz = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 1]], ones), axis=2))
    c = np.linalg.det(np.concatenate((normsq, tetrapos), axis=2))
    r = np.sqrt(Dx ** 2 + Dy ** 2 + Dz ** 2 - 4 * a * c) / (2 * np.abs(a))
    # Find tetrahedrals
    tetras = tetra.vertices[r < alpha, :]
    # triangles
    TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    Triangles = tetras[:, TriComb].reshape(-1, 3)
    Triangles = np.sort(Triangles, axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = defaultdict(int)
    for tri in Triangles: TrianglesDict[tuple(tri)] += 1
    Triangles = np.array([tri for tri in TrianglesDict if TrianglesDict[tri] == 1])
    # edges
    EdgeComb = np.array([(0, 1), (0, 2), (1, 2)])
    Edges = Triangles[:, EdgeComb].reshape(-1, 2)
    Edges = np.sort(Edges, axis=1)
    Edges = np.unique(Edges, axis=0)

    Vertices = np.unique(Edges)
    return Vertices, Edges, Triangles

def concave_hull(coords, shape, alpha=5):
    verts, faces, triangles = alpha_shape_3D(coords, alpha=alpha)

    f = []
    for t in triangles:
        f.append(np.stack((coords[t[0]], coords[t[1]], coords[t[2]])))
    f = np.stack(f)

    alpha_vol = np.zeros(shape)
    for z, y, x in voxelize(np.stack(f)):
        alpha_vol[z, y, x] = 1
    return alpha_vol.astype(int), binary_fill_holes(alpha_vol).astype(int)

def hausdorff_pair(image0, image1):
    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))

    # If either of the sets are empty, there is no corresponding pair of points
    if len(a_points) == 0 or len(b_points) == 0:
        return (), ()

    nearest_dists_from_b, nearest_a_point_indices_from_b = cKDTree(a_points).query(b_points)
    nearest_dists_from_a, nearest_b_point_indices_from_a = cKDTree(b_points).query(a_points)

    max_index_from_a = nearest_dists_from_b.argmax()
    max_index_from_b = nearest_dists_from_a.argmax()

    max_dist_from_a = nearest_dists_from_b[max_index_from_a]
    max_dist_from_b = nearest_dists_from_a[max_index_from_b]

    if max_dist_from_b > max_dist_from_a:
        return a_points[max_index_from_b], b_points[nearest_b_point_indices_from_a[max_index_from_b]]
    else:
        return a_points[nearest_a_point_indices_from_b[max_index_from_a]], b_points[max_index_from_a]