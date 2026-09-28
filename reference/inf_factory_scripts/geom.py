"""Minimal mesh primitives (numpy only) + binary PLY writer. Copied-not-imported by convention
from ../indoor-lab-scene/PLAN.md: walls/floor are single planes, clutter are closed solids."""
import struct
import numpy as np


def write_ply(path, verts, faces):
    verts = np.asarray(verts, dtype="<f4")
    faces = np.asarray(faces, dtype="<u4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\nproperty list uchar uint vertex_indices\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes())
        for tri in faces:
            f.write(struct.pack("<B3I", 3, *tri))


def quad(p0, p1, p2, p3):
    """Rectangle from 4 corners in CCW order (normal by right-hand rule) -> 2 triangles."""
    return np.array([p0, p1, p2, p3], float), np.array([[0, 1, 2], [0, 2, 3]])


def box(center_xy, size_xy, height, z0=0.0, yaw_deg=0.0):
    """Closed cuboid, outward normals. size_xy=(dx,dy) footprint, height along z."""
    cx, cy = center_xy
    hx, hy = size_xy[0] / 2, size_xy[1] / 2
    c, s = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    R = np.array([[c, -s], [s, c]])
    fp = (R @ np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]).T).T + [cx, cy]
    v = np.vstack([np.c_[fp, np.full(4, z0)], np.c_[fp, np.full(4, z0 + height)]])
    f = [[0, 2, 1], [0, 3, 2],            # bottom (down)
         [4, 5, 6], [4, 6, 7],            # top (up)
         [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
         [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    return v, np.array(f)


def cylinder(center_xy, radius, height, z0=0.0, n_seg=24):
    """Closed vertical cylinder (n_seg-gon prism), outward normals."""
    cx, cy = center_xy
    a = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    ring = np.c_[cx + radius * np.cos(a), cy + radius * np.sin(a)]
    v = np.vstack([np.c_[ring, np.full(n_seg, z0)], np.c_[ring, np.full(n_seg, z0 + height)],
                   [[cx, cy, z0]], [[cx, cy, z0 + height]]])
    bc, tc = 2 * n_seg, 2 * n_seg + 1
    f = []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        f += [[i, j, n_seg + j], [i, n_seg + j, n_seg + i],   # side
               [bc, j, i],                                      # bottom cap
               [tc, n_seg + i, n_seg + j]]                      # top cap
    return v, np.array(f)
