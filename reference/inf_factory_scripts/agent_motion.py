"""Continuous random-waypoint movers (AGVs + UE) on the free-space grid, for the multi-agent CIR run (2026-09-22).
Free-space grid / A* / smoothing logic is a direct copy of plan_ue_paths.py's proven algorithm (that script's
version is argparse-global, not reusable as a module - copied rather than refactored, per the 2026-09-22 design
review). Motion model idea (per-trip randomized speed, dwell time at destination, random waypoint selection)
adapted from the earlier four-room project (~/wisegrt-eval/scripts/coverage_marl_env.py) - simplified since this
hall has no rooms/doors/task queue. Speed range, dwell-time distribution: MY ASSUMPTIONS, confirmed with the user
2026-09-22 (AGV speed ~ Uniform(0.5x, 1.0x) of spec max per trip; AGV dwell ~ Exponential(mean 7.5s); UE: constant
0.5 m/s, no dwell). No collision avoidance between movers (matches the four-room project's precedent)."""
import json
import numpy as np
import heapq


class FreeSpaceGrid:
    def __init__(self, scene_dir, clear=0.5, h=0.1, L=120.0, W=60.0):
        self.L, self.W, self.h, self.clear = L, W, h, clear
        self.cl = json.load(open(f"{scene_dir}/clutter.json"))["objects"]
        gx = np.arange(0, L + 1e-9, h); gy = np.arange(0, W + 1e-9, h)
        self.gx, self.gy = gx, gy
        GX, GY = np.meshgrid(gx, gy)
        F = self.free(GX, GY)
        lab = np.zeros(F.shape, int); nlab = 0
        for iy, ix in zip(*np.where(F)):
            if lab[iy, ix]: continue
            nlab += 1; stack = [(iy, ix)]; lab[iy, ix] = nlab
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < F.shape[0] and 0 <= nx < F.shape[1] and F[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = nlab; stack.append((ny, nx))
        sizes = np.bincount(lab.ravel())[1:]; main = int(np.argmax(sizes)) + 1
        self.M = lab == main
        self.cand = np.argwhere(self.M)   # (iy, ix) rows in the main reachable component

    def clearance(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        d = np.minimum.reduce([x, self.L - x, y, self.W - y])
        for o in self.cl:
            if o["type"] == "box":
                qx = abs(x - o["cx"]) - o["dx"] / 2; qy = abs(y - o["cy"]) - o["dy"] / 2
                dd = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)
            else:
                dd = np.hypot(x - o["cx"], y - o["cy"]) - o["r"]
            d = np.minimum(d, dd)
        return d

    def free(self, x, y):
        return self.clearance(x, y) >= self.clear

    def seg_free(self, p, q, step=0.02):
        n = max(int(np.hypot(*(q - p)) / step), 2); t = np.linspace(0, 1, n)
        return bool(self.free(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])).all())

    def astar(self, s, g):
        h = self.h
        S = (int(round(s[1] / h)), int(round(s[0] / h))); G = (int(round(g[1] / h)), int(round(g[0] / h)))
        dist = {S: 0.0}; prev = {}; pq = [(0.0, S)]
        nb = [(dy, dx, np.hypot(dy, dx)) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
        while pq:
            _, u = heapq.heappop(pq)
            if u == G: break
            du = dist[u]
            for dy, dx, c in nb:
                v = (u[0] + dy, u[1] + dx)
                if not (0 <= v[0] < self.M.shape[0] and 0 <= v[1] < self.M.shape[1]) or not self.M[v]: continue
                if dy and dx and not (self.M[u[0] + dy, u[1]] and self.M[u[0], u[1] + dx]): continue
                nd = du + c
                if nd < dist.get(v, 1e18):
                    dist[v] = nd; prev[v] = u
                    heapq.heappush(pq, (nd + np.hypot(v[0] - G[0], v[1] - G[1]), v))
        path = [G]
        while path[-1] != S: path.append(prev[path[-1]])
        return np.array([[self.gx[ix], self.gy[iy]] for iy, ix in path[::-1]])

    def smooth(self, P):
        out = [P[0]]; i = 0
        while i < len(P) - 1:
            j = len(P) - 1
            while j > i + 1 and not self.seg_free(P[i], P[j]): j -= 1
            out.append(P[j]); i = j
        return np.array(out)

    def sample_point(self, rng):
        iy, ix = self.cand[rng.integers(len(self.cand))]
        return np.array([self.gx[ix], self.gy[iy]])

    def nearest_free(self, p):
        d2 = (self.gx[self.cand[:, 1]] - p[0]) ** 2 + (self.gy[self.cand[:, 0]] - p[1]) ** 2
        iy, ix = self.cand[np.argmin(d2)]
        return np.array([self.gx[ix], self.gy[iy]])

    def route(self, p, dst):
        """Smoothed path from continuous point p (snapped to the nearest free grid node for A*) to dst,
        with the actual starting point p prepended so motion stays continuous (no snap jump)."""
        snapped = self.nearest_free(p)
        P = self.smooth(self.astar(snapped, dst))
        if np.linalg.norm(p - P[0]) > 1e-9:
            P = np.vstack([p, P])
        return P


def random_waypoint_goal(pos, rng, grid, state):
    """Mode A goal picker: uniform random point in the reachable free area (unchanged from the original model)."""
    return grid.sample_point(rng)


def make_corner_diagonal_goal(grid, jitter=1.0):
    """Mode B goal picker (2026-09-23), matching the LiDAR-beam-prediction paper's ORCA setup as closely as this
    scene allows: corners as start/destination, always the DIAGONALLY opposite corner (not adjacent) - the paper's
    own text ("opposite side of the area") plus the user's explicit instruction. A small random jitter around the
    exact corner point (clamped inside the hall, snapped to free space) is OUR addition, not in the paper - avoids
    perfectly deterministic point-repetition every leg. Call `mover.state['corner_idx'] = i` before the first trip
    to assign a mover to one of the 4 corners as its start; otherwise the nearest corner to its initial position
    is used."""
    corners = np.array([[0.0, 0.0], [grid.L, 0.0], [0.0, grid.W], [grid.L, grid.W]])
    diag_of = {0: 3, 3: 0, 1: 2, 2: 1}   # (0,0)<->(L,W) and (L,0)<->(0,W): always the diagonal, never adjacent

    def goal_picker(pos, rng, grid_, state):
        if "corner_idx" not in state:
            d = np.linalg.norm(corners - pos, axis=1)
            state["corner_idx"] = int(np.argmin(d))
        state["corner_idx"] = diag_of[state["corner_idx"]]
        target = corners[state["corner_idx"]] + rng.uniform(-jitter, jitter, size=2)
        target = np.clip(target, [0.6, 0.6], [grid_.L - 0.6, grid_.W - 0.6])
        return grid_.nearest_free(target)

    return goal_picker


class PathMover:
    """Waypoint-pointer path follower (2026-09-23) - like Mover, but exposes a preferred_velocity()/advance() pair
    instead of self-integrating, so an external physics step (ORCA collision avoidance) can drive the actual
    position each tick. `goal_picker(pos, rng, grid, state)` returns the next destination point; `state` is a
    per-mover dict the picker may use for its own bookkeeping (e.g. Mode B's current corner index)."""

    def __init__(self, grid, rng, pos0, speed_fn, dwell_fn, goal_picker, init_state=None):
        self.grid = grid; self.rng = rng; self.pos = np.array(pos0, float)
        self.speed_fn = speed_fn; self.dwell_fn = dwell_fn; self.goal_picker = goal_picker
        self.state = dict(init_state or {})
        self.dwell_remaining = 0.0
        self._new_trip()

    def _new_trip(self):
        dst = self.goal_picker(self.pos, self.rng, self.grid, self.state)
        self.path = self.grid.route(self.pos, dst)
        self.wp_idx = 1 if len(self.path) > 1 else 0
        self.speed = self.speed_fn(self.rng)

    def preferred_velocity(self):
        """What this mover would do with no other agents around - ORCA adjusts this for collision avoidance."""
        if self.dwell_remaining > 0:
            return np.zeros(2)
        target = self.path[min(self.wp_idx, len(self.path) - 1)]
        d = target - self.pos; dist = np.linalg.norm(d)
        return d / dist * self.speed if dist > 1e-6 else np.zeros(2)

    def advance(self, dt, new_pos):
        """Called once per tick with the agent's actual (collision-aware) new position from the ORCA step."""
        self.pos = new_pos
        if self.dwell_remaining > 0:
            self.dwell_remaining -= dt
            if self.dwell_remaining <= 0: self._new_trip()
            return
        while self.wp_idx < len(self.path) - 1 and np.linalg.norm(self.path[self.wp_idx] - self.pos) < 0.3:
            self.wp_idx += 1
        if self.wp_idx >= len(self.path) - 1 and np.linalg.norm(self.path[-1] - self.pos) < 0.3:
            dw = self.dwell_fn(self.rng)
            if dw > 0: self.dwell_remaining = dw
            else: self._new_trip()


class OrcaGroup:
    """Wraps RVO2 (github.com/sybrenstuvel/Python-RVO2, installed 2026-09-23) to add real agent-agent collision
    avoidance on top of PathMover's A*-planned routes - the gap flagged all session ("no collision avoidance
    between movers"). Static clutter is NOT given to ORCA as obstacles (hybrid design, confirmed with the user):
    A*+LOS-smoothing already routes each mover's plan around it; ORCA here only adjusts velocity to avoid other
    moving agents reciprocally."""

    def __init__(self, dt, movers, radii, max_speeds, neighbor_dist=15.0, max_neighbors=10, time_horizon=3.0):
        import rvo2
        self.dt = dt; self.movers = movers
        self.sim = rvo2.PyRVOSimulator(dt, neighbor_dist, max_neighbors, time_horizon, time_horizon, 0.3, 1.0)
        self.agent_ids = [
            self.sim.addAgent(tuple(m.pos), neighbor_dist, max_neighbors, time_horizon, time_horizon, r, v, (0.0, 0.0))
            for m, r, v in zip(movers, radii, max_speeds)
        ]

    def step(self):
        for aid, m in zip(self.agent_ids, self.movers):
            self.sim.setAgentPrefVelocity(aid, tuple(m.preferred_velocity()))
        self.sim.doStep()
        for aid, m in zip(self.agent_ids, self.movers):
            m.advance(self.dt, np.array(self.sim.getAgentPosition(aid)))


class Mover:
    """Continuous random-waypoint mover with per-trip speed and optional dwell time at each destination."""

    def __init__(self, grid, rng, pos0, speed_fn, dwell_fn):
        self.grid = grid; self.rng = rng; self.pos = np.array(pos0, float)
        self.speed_fn = speed_fn; self.dwell_fn = dwell_fn
        self.dwell_remaining = 0.0
        self._new_trip()

    def _new_trip(self):
        dst = self.grid.sample_point(self.rng)
        self.path = self.grid.route(self.pos, dst)
        seg = np.hypot(*np.diff(self.path, axis=0).T)
        self.cum = np.r_[0, np.cumsum(seg)]
        self.dist_along = 0.0
        self.speed = self.speed_fn(self.rng)

    def step(self, dt):
        if self.dwell_remaining > 0:
            self.dwell_remaining -= dt
            if self.dwell_remaining <= 0: self._new_trip()
            return self.pos
        self.dist_along += self.speed * dt
        total = self.cum[-1]
        if self.dist_along >= total:
            self.pos = self.path[-1].copy()
            dw = self.dwell_fn(self.rng)
            if dw > 0: self.dwell_remaining = dw
            else: self._new_trip()
            return self.pos
        x = np.interp(self.dist_along, self.cum, self.path[:, 0])
        y = np.interp(self.dist_along, self.cum, self.path[:, 1])
        self.pos = np.array([x, y])
        return self.pos
