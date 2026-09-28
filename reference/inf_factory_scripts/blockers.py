"""Moving blockers (TR 38.901 Table 7.6.4.2-5, Clause 7.6.4.2 "Blockage model B").
Spec gives width/height only (flat screen, analytical knife-edge model) - we build solid 3D mesh boxes instead
(2026-09-22 decision) so the existing PathSolver-based pipeline handles reflection/diffraction from them directly.
depth and material are NOT spec values (spec is silent on both for a solid object) - marked as assumptions below.
"""
import os, numpy as np
from geom import box, write_ply
import sionna.rt as rt

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "blockers")

# w, h = TR 38.901 Table 7.6.4.2-5 (spec). depth, material = assumptions (spec gives neither for a solid object),
# confirmed with the user 2026-09-22. max_speed_mps = spec's "up to" mobility figure.
# agv h: spec value is 1.5 m, which exactly equals the TX/UE height (1.5 m) used throughout this study, so the
# direct LOS ray grazes the box's top edge instead of being blocked (confirmed empirically 2026-09-22, pilot
# results/pilot_agv_crossing - first-arrival path power was unchanged even at the exact crossing point). Per the
# user's explicit request, h is overridden to 2.0 m here to give real intersection margin. This is NOT the spec
# value for AGV height - it's a deliberate deviation for the blocking pilot only.
BLOCKER_SPECS = {
    "human": dict(w=0.3, h=1.7, depth=0.3, itu_type="wet_ground", max_speed_mps=3 / 3.6),
    "agv":   dict(w=3.0, h=2.0, depth=1.5, itu_type="metal",      max_speed_mps=30 / 3.6),
    "robot": dict(w=2.0, h=0.2, depth=0.5, itu_type="metal",      max_speed_mps=3.0),
}


def blocker_mesh_path(kind):
    return os.path.join(ASSET_DIR, f"{kind}.ply")


def build_blocker_mesh(kind):
    """Box centred at local (0,0), floor at local z=0, so SceneObject.position places the FLOOR CENTRE at world (x,y,0)."""
    spec = BLOCKER_SPECS[kind]
    os.makedirs(ASSET_DIR, exist_ok=True)
    v, f = box(center_xy=(0, 0), size_xy=(spec["w"], spec["depth"]), height=spec["h"], z0=0.0)
    path = blocker_mesh_path(kind)
    write_ply(path, v, f)
    return path


def make_blocker(kind, name=None):
    """Returns a SceneObject for the given blocker kind, NOT yet added to any scene (caller: scene.edit(add=...))."""
    spec = BLOCKER_SPECS[kind]
    path = blocker_mesh_path(kind)
    if not os.path.exists(path): build_blocker_mesh(kind)
    obj_name = name or kind
    mat = rt.ITURadioMaterial(name=f"{obj_name}-material", itu_type=spec["itu_type"], thickness=0.01)
    mesh = rt.load_mesh(path)
    return rt.SceneObject(mi_mesh=mesh, name=obj_name, radio_material=mat)


def floor_position(kind, x, y):
    """mi.Point3f for placing a blocker's FLOOR at world (x, y, 0).
    SceneObject.position is the object's AABB CENTRE, not its local mesh origin (confirmed 2026-09-22 by reading
    sionna/rt/scene_object.py) - since build_blocker_mesh() builds the mesh spanning local z=[0, h], the AABB
    centre sits at local z=h/2. Setting position=(x, y, 0) silently puts the box HALF BELOW THE FLOOR
    (z=[-h/2, +h/2]) instead of resting on it - this caused two failed blocking pilots before being caught."""
    import mitsuba as mi
    return mi.Point3f(float(x), float(y), BLOCKER_SPECS[kind]["h"] / 2)
