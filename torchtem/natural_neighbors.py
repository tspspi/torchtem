from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import ConvexHull, Delaunay, cKDTree


def triangle_area(pt1: np.ndarray, pt2: np.ndarray, pt3: np.ndarray) -> float:
    a = (
        pt1[0] * pt2[1]
        - pt2[0] * pt1[1]
        + pt2[0] * pt3[1]
        - pt3[0] * pt2[1]
        + pt3[0] * pt1[1]
        - pt1[0] * pt3[1]
    )
    return abs(a) / 2.0


def circumcircle_radius(pt0: np.ndarray, pt1: np.ndarray, pt2: np.ndarray) -> float:
    a = np.linalg.norm(pt0 - pt1)
    b = np.linalg.norm(pt1 - pt2)
    c = np.linalg.norm(pt2 - pt0)
    area = triangle_area(pt0, pt1, pt2)
    if area > 0:
        return (a * b * c) / (4 * area)
    return np.nan


def circumcenter(pt0: np.ndarray, pt1: np.ndarray, pt2: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a_x, a_y = pt0
    b_x, b_y = pt1
    c_x, c_y = pt2
    bc_y_diff = b_y - c_y
    ca_y_diff = c_y - a_y
    ab_y_diff = a_y - b_y
    cb_x_diff = c_x - b_x
    ac_x_diff = a_x - c_x
    ba_x_diff = b_x - a_x
    d_div = a_x * bc_y_diff + b_x * ca_y_diff + c_x * ab_y_diff
    if abs(d_div) < eps:
        return np.mean([pt0, pt1, pt2], axis=0)
    d_inv = 0.5 / d_div
    a_mag = a_x * a_x + a_y * a_y
    b_mag = b_x * b_x + b_y * b_y
    c_mag = c_x * c_x + c_y * c_y
    cx = (a_mag * bc_y_diff + b_mag * ca_y_diff + c_mag * ab_y_diff) * d_inv
    cy = (a_mag * cb_x_diff + b_mag * ac_x_diff + c_mag * ba_x_diff) * d_inv
    return np.array([cx, cy])


def find_natural_neighbors(tri: Delaunay, grid_points: np.ndarray):
    tree = cKDTree(grid_points)
    triangle_info = []
    members = {key: [] for key in range(len(tree.data))}
    for i, indices in enumerate(tri.simplices):
        triangle = tri.points[indices]
        cc = circumcenter(*triangle)
        r = circumcircle_radius(*triangle)
        triangle_info.append(cc)
        for point in tree.query_ball_point(cc, r):
            members[point].append(i)
    return members, np.array(triangle_info)


def find_local_boundary(tri: Delaunay, triangles: list[int]) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for triangle in triangles:
        for i in range(3):
            pt1 = tri.simplices[triangle][i]
            pt2 = tri.simplices[triangle][(i + 1) % 3]
            if (pt1, pt2) in edges:
                edges.remove((pt1, pt2))
            elif (pt2, pt1) in edges:
                edges.remove((pt2, pt1))
            else:
                edges.append((pt1, pt2))
    return edges


def polygon_area(poly: list[np.ndarray]) -> float:
    a = 0.0
    n = len(poly)
    for i in range(n):
        a += poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
    return abs(a) / 2.0


def order_edges(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    edge = edges[0]
    edges = edges[1:]
    ordered_edges = [edge]
    num_max = len(edges)
    while len(edges) > 0 and num_max > 0:
        match = edge[1]
        for search_edge in edges:
            if match == search_edge[0]:
                edge = search_edge
                edges.remove(edge)
                ordered_edges.append(search_edge)
                break
        num_max -= 1
    return ordered_edges


def natural_neighbor_weights(
    points: np.ndarray,
    query_point: np.ndarray,
    tri: Delaunay,
    neighbors: list[int],
    circumcenters: np.ndarray,
) -> np.ndarray:
    weights = np.zeros(len(points), dtype=np.float64)
    overlap = np.isclose(query_point[0], points[:, 0]) * np.isclose(query_point[1], points[:, 1])
    if np.any(overlap):
        weights[np.where(overlap)[0]] = 1.0
        return weights

    edges = find_local_boundary(tri, neighbors)
    edge_vertices = [segment[0] for segment in order_edges(edges)]
    num_vertices = len(edge_vertices)
    p1 = edge_vertices[0]
    p2 = edge_vertices[1]
    c1 = circumcenter(query_point, tri.points[p1], tri.points[p2])
    polygon = [c1]
    for i in range(num_vertices):
        p3 = edge_vertices[(i + 2) % num_vertices]
        c2 = circumcenter(query_point, tri.points[p3], tri.points[p2])
        polygon.append(c2)
        for check_tri in neighbors:
            if p2 in tri.simplices[check_tri]:
                polygon.append(circumcenters[check_tri])
        pts = [polygon[i] for i in ConvexHull(polygon).vertices]
        area = polygon_area(pts)
        weights[
            (tri.points[p2][0] == points[:, 0]) & (tri.points[p2][1] == points[:, 1])
        ] += area
        polygon = [c2]
        p2 = p3
    return weights / weights.sum()


def pairwise_natural_neighbor_weights(points: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
    points_np = points.detach().cpu().numpy()
    query_np = query_points.detach().cpu().numpy()
    triangulation = Delaunay(points_np)
    members, circumcenters = find_natural_neighbors(triangulation, query_np)
    weights = np.zeros((len(points_np), len(query_np)), dtype=np.float64)
    for i, query_point in enumerate(query_np):
        weights[:, i] = natural_neighbor_weights(
            points_np, query_point, triangulation, members[i], circumcenters
        )
    return torch.tensor(weights, dtype=torch.float64, device=points.device)
