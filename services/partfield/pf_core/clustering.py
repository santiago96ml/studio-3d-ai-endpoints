"""Pure-SciPy/scikit-learn mesh feature clustering (replaces Open3D/NetworkX/PyMeshLab bloat)."""

from __future__ import annotations

from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors


def construct_connected_face_adjacency(faces: np.ndarray, vertices: np.ndarray, k: int = 8) -> sp.csr_matrix:
    """Builds shared-edge face adjacency guaranteed to form a single connected component via SciPy MST."""
    num_faces = len(faces)
    if num_faces == 0:
        return sp.csr_matrix((0, 0), dtype=np.int8)

    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges.sort(axis=1)
    edge_ids = edges[:, 0].astype(np.int64) * (1 << 32) + edges[:, 1].astype(np.int64)
    face_ids = np.tile(np.arange(num_faces, dtype=np.int64), 3)
    order = np.argsort(edge_ids, kind="stable")
    es, fs = edge_ids[order], face_ids[order]
    shared = np.flatnonzero(es[:-1] == es[1:])
    row = np.r_[fs[shared], fs[shared + 1]]
    col = np.r_[fs[shared + 1], fs[shared]]

    adj = sp.coo_matrix((np.ones(len(row), dtype=np.int8), (row, col)), shape=(num_faces, num_faces)).tocsr()
    n_comp, comp_labels = connected_components(adj, directed=False)
    if n_comp <= 1:
        return adj

    centroids = vertices[faces].mean(axis=1)
    comp_rep = np.array([int(np.flatnonzero(comp_labels == cid)[0]) for cid in range(n_comp)], dtype=np.int64)
    comp_cent = np.array([centroids[comp_labels == cid].mean(axis=0) for cid in range(n_comp)], dtype=np.float64)

    n_nbrs = min(k, n_comp)
    nbrs = NearestNeighbors(n_neighbors=n_nbrs, algorithm="auto").fit(comp_cent)
    dists, idxs = nbrs.kneighbors(comp_cent)
    r_c = np.repeat(np.arange(n_comp), n_nbrs)
    c_c = idxs.reshape(-1)
    d_c = np.maximum(dists.reshape(-1), 1e-8)
    valid = r_c != c_c
    mst = minimum_spanning_tree(sp.coo_matrix((d_c[valid], (r_c[valid], c_c[valid])), shape=(n_comp, n_comp)))
    mst_coo = mst.tocoo()

    add_r = np.r_[comp_rep[mst_coo.row], comp_rep[mst_coo.col]]
    add_c = np.r_[comp_rep[mst_coo.col], comp_rep[mst_coo.row]]
    if len(add_r) > 0:
        adj = (adj + sp.coo_matrix((np.ones(len(add_r), dtype=np.int8), (add_r, add_c)), shape=(num_faces, num_faces))).tocsr()
    return adj


def cluster_part_features(
    face_features: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    num_parts: int = 10,
    min_part_rel_area: float = 0.008,
) -> np.ndarray:
    """Clusters normalized per-face PartField vectors into cohesive 3D semantic parts."""
    n_faces = len(faces)
    if n_faces == 0:
        return np.zeros(0, dtype=np.int32)
    if n_faces <= max(2, num_parts):
        return np.arange(n_faces, dtype=np.int32)

    norms = np.linalg.norm(face_features, axis=-1, keepdims=True)
    feats_unit = face_features / np.maximum(norms, 1e-8)

    # Para mallas densas (p.ej. 50k-150k caras de Hunyuan3D-2), contraer en superparches espaciales
    # conectados (~8,000 celdas) acelera AgglomerativeClustering de ~30s a <0.5s manteniendo bordes limpios.
    if n_faces > 10000:
        from scipy.spatial import cKDTree

        centroids = vertices[faces].mean(axis=1)
        rng = np.random.default_rng(42)
        n_seeds = 8000
        seed_idx = rng.choice(n_faces, size=n_seeds, replace=False)
        _, cell_id = cKDTree(centroids[seed_idx]).query(centroids, workers=1)
        _, cell_id = np.unique(cell_id, return_inverse=True)
        n_cells = int(cell_id.max()) + 1

        cell_feats = np.zeros((n_cells, feats_unit.shape[1]), dtype=np.float64)
        np.add.at(cell_feats, cell_id, feats_unit)
        cell_counts = np.bincount(cell_id, minlength=n_cells).astype(np.float64)[:, None]
        cell_feats = cell_feats / np.maximum(cell_counts, 1.0)
        cell_feats = cell_feats / np.maximum(np.linalg.norm(cell_feats, axis=-1, keepdims=True), 1e-8)
        cell_cent = np.zeros((n_cells, 3), dtype=np.float64)
        np.add.at(cell_cent, cell_id, centroids)
        cell_cent = cell_cent / np.maximum(cell_counts, 1.0)

        # Adyacencia entre superparches usando bordes compartidos de la malla original
        raw_adj = construct_connected_face_adjacency(faces, vertices).tocoo()
        ca, cb = cell_id[raw_adj.row], cell_id[raw_adj.col]
        diff = ca != cb
        cell_adj = sp.coo_matrix((np.ones(int(diff.sum()), dtype=np.int8), (ca[diff], cb[diff])), shape=(n_cells, n_cells)).tocsr()
        n_cc, cc_lab = connected_components(cell_adj, directed=False)
        if n_cc > 1:
            reps = np.array([int(np.flatnonzero(cc_lab == cid)[0]) for cid in range(n_cc)], dtype=np.int64)
            rc_cent = np.array([cell_cent[cc_lab == cid].mean(axis=0) for cid in range(n_cc)], dtype=np.float64)
            nbrs = NearestNeighbors(n_neighbors=min(8, n_cc), algorithm="auto").fit(rc_cent)
            dists, idxs = nbrs.kneighbors(rc_cent)
            rr = np.repeat(np.arange(n_cc), min(8, n_cc))
            cc_i = idxs.reshape(-1)
            dd = np.maximum(dists.reshape(-1), 1e-8)
            v_m = rr != cc_i
            mst = minimum_spanning_tree(sp.coo_matrix((dd[v_m], (rr[v_m], cc_i[v_m])), shape=(n_cc, n_cc))).tocoo()
            ar = np.r_[reps[mst.row], reps[mst.col]]
            ac = np.r_[reps[mst.col], reps[mst.row]]
            if len(ar) > 0:
                cell_adj = (cell_adj + sp.coo_matrix((np.ones(len(ar), dtype=np.int8), (ar, ac)), shape=(n_cells, n_cells))).tocsr()

        k = max(2, min(int(num_parts), n_cells - 1))
        cell_labels = AgglomerativeClustering(connectivity=cell_adj, n_clusters=k, linkage="ward").fit_predict(cell_feats)
        labels = cell_labels[cell_id].astype(np.int32)
    else:
        adj = construct_connected_face_adjacency(faces, vertices)
        k = max(2, min(int(num_parts), n_faces - 1))
        labels = AgglomerativeClustering(connectivity=adj, n_clusters=k, linkage="ward").fit_predict(feats_unit).astype(np.int32)

    # Absorb microscopic face specks inside each semantic part via shared boundary length
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    min_area = float(areas.sum()) * min_part_rel_area
    if min_area > 0:
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
        e_len = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
        edges.sort(axis=1)
        e_ids = edges[:, 0].astype(np.int64) * (1 << 32) + edges[:, 1].astype(np.int64)
        f_ids = np.tile(np.arange(n_faces, dtype=np.int64), 3)
        order = np.argsort(e_ids, kind="stable")
        es, fs, ls = e_ids[order], f_ids[order], e_len[order]
        sh = np.flatnonzero(es[:-1] == es[1:])
        ea, eb, el = fs[sh], fs[sh + 1], ls[sh]

        for _ in range(3):
            same = labels[ea] == labels[eb]
            g = sp.coo_matrix((np.ones(int(same.sum()), dtype=np.int8), (ea[same], eb[same])), shape=(n_faces, n_faces))
            nc, cc = connected_components(g, directed=False)
            c_area = np.bincount(cc, weights=areas, minlength=nc)
            bad = np.flatnonzero(c_area < min_area)
            if len(bad) == 0:
                break
            bnd = ~same
            if not np.any(bnd):
                break
            ca, cb, ln = cc[ea[bnd]], cc[eb[bnd]], el[bnd]
            best_len: dict[tuple[int, int], float] = defaultdict(float)
            for s_c, d_l, w in zip(ca, labels[eb[bnd]], ln):
                best_len[(int(s_c), int(d_l))] += float(w)
            for s_c, d_l, w in zip(cb, labels[ea[bnd]], ln):
                best_len[(int(s_c), int(d_l))] += float(w)
            bad_set = set(int(x) for x in bad)
            winners = {}
            for (sc, dl), w in best_len.items():
                if sc in bad_set and w > winners.get(sc, (-1.0, -1))[0]:
                    winners[sc] = (w, dl)
            if not winners:
                break
            for sc, (_, dl) in winners.items():
                labels[cc == sc] = dl

    _, compact = np.unique(labels, return_inverse=True)
    return compact.astype(np.int32)
