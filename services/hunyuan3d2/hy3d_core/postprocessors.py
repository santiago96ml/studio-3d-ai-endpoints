# Copyright (C) 2024 Tencent / THL A29 Limited.
# Licensed under the TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT (see ../LICENSE-HUNYUAN3D.txt).
# Distilled pure-RAM postprocessing (trimesh + fast-simplification + manifold check), zero pymeshlab/tempfiles.

import numpy as np
import trimesh


def count_non_manifold_edges(mesh: trimesh.Trimesh) -> int:
    if len(mesh.faces) == 0:
        return 0
    edges = mesh.edges_sorted
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts != 2))


def remove_floaters(mesh: trimesh.Trimesh, min_face_ratio: float = 0.005) -> trimesh.Trimesh:
    """Drop tiny disconnected mesh islands while always retaining the largest component."""
    if len(mesh.faces) == 0:
        return mesh
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    total_faces = sum(len(p.faces) for p in parts)
    threshold = max(int(total_faces * min_face_ratio), 1)
    parts_sorted = sorted(parts, key=lambda p: len(p.faces), reverse=True)
    kept = [parts_sorted[0]] + [p for p in parts_sorted[1:] if len(p.faces) >= threshold]
    if len(kept) == 1:
        return kept[0]
    return trimesh.util.concatenate(kept)


def clean_mesh_in_memory(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.merge_vertices(merge_tex=False, merge_norm=False)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_normals(mesh)
    return mesh


def decimate_mesh_in_memory(mesh: trimesh.Trimesh, max_facenum: int = 150000) -> trimesh.Trimesh:
    if max_facenum <= 0 or len(mesh.faces) <= max_facenum:
        return mesh
    try:
        decimated = mesh.simplify_quadric_decimation(face_count=int(max_facenum))
        return clean_mesh_in_memory(decimated)
    except ImportError:
        return mesh


def repair_watertight_if_needed(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Close small boundary loops / validate 2-manifold topology in memory."""
    if len(mesh.faces) == 0:
        return mesh

    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)

    # Optional manifold3d canonicalization pass if the mesh is already clean enough for Manifold
    try:
        import manifold3d

        m_in = manifold3d.Mesh(
            vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
            tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
        )
        mf = manifold3d.Manifold(m_in)
        m_out = mf.to_mesh()
        repaired = trimesh.Trimesh(
            vertices=np.asarray(m_out.vert_properties[:, :3], dtype=np.float32),
            faces=np.asarray(m_out.tri_verts, dtype=np.int64),
            process=True,
        )
        if len(repaired.faces) > 0 and repaired.is_watertight:
            return repaired
    except Exception:
        pass

    return mesh


def clean_and_repair_mesh(
    mesh: trimesh.Trimesh,
    max_faces: int = 150000,
    min_floater_ratio: float = 0.005,
    ensure_watertight: bool = True,
) -> tuple[trimesh.Trimesh, dict]:
    """Run in-memory floater removal, degenerate face cleanup, QEM decimation and manifold check."""
    cleaned = clean_mesh_in_memory(mesh)
    cleaned = remove_floaters(cleaned, min_face_ratio=min_floater_ratio)
    if max_faces and len(cleaned.faces) > max_faces:
        cleaned = decimate_mesh_in_memory(cleaned, max_facenum=max_faces)
    if ensure_watertight:
        cleaned = repair_watertight_if_needed(cleaned)

    stats = {
        "vertices": int(len(cleaned.vertices)),
        "faces": int(len(cleaned.faces)),
        "watertight": bool(cleaned.is_watertight),
        "non_manifold_edges": count_non_manifold_edges(cleaned),
        "components": int(len(cleaned.split(only_watertight=False))) if len(cleaned.faces) > 0 else 0,
    }
    return cleaned, stats
