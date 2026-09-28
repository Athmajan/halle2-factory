"""Dec-POMDP environment: 2 collaborative coverage (TX) agents + 4 UEs doing a
task-driven box-relocation job in the 4-room grid, backed entirely by the
precomputed path-gain cache (no live Sionna calls per step - see
precompute_tx_radio_map_cache.py).

Continuing task (no episode boundaries): boxes arrive via a Poisson process at
random points along the LEFT wall (rooms A+B) and must be carried to one of a
fixed set of goal points along the RIGHT wall (rooms B'+A'), drawn from a
DIFFERENT (discrete, non-uniform) distribution than the pickup side - every
trip is spatially unpredictable at both ends, not just noisy in between. Path
*shape* is also randomized (per-trip door choice + Gauss-Markov wander, not a
straight line) so a coverage agent cannot simply memorize one fixed corridor -
see ../../ocudu/tasks/multimodal-scheduler-comparison/
TASK_DRIVEN_UE_MOBILITY_AND_ANTICIPATORY_WORLD_MODEL_PLACEMENT.md for the
design rationale on replacing a memoryless random walk with a structured-but-
unpredictable task. Boxes are one of two types - "small" (faster to carry,
quicker to unload) and "big" (slower, slower to unload) - each drawing its
carry speed and unload/wait time from a type-specific *range* per trip. A UE
also has a battery that drains continuously and, once it drops to 10%, forces
a detour to one of 8 charging stations (4 fast/"top", 4 slow/"bottom", picked
uniformly at random) where it charges at a station-specific rate until full,
then resumes exactly the task it was interrupted from. A UE also halts in
place - mid-route, whether carrying a box, returning empty, or traveling to/
from a charging station - whenever its instantaneous SNR drops below the
minimum operating threshold: this is the direct physical link between the TX
coverage agents' job and task throughput. None of this pauses box arrivals -
the Poisson process runs unconditionally every tick regardless of any UE's
state. TX agents act on a slower cadence than UEs. Partial observability: each
TX agent only observes the UEs currently on its own side (agent1: rooms A+B,
agent2: rooms B'+A'). Reward is the shared team objective: average SNR (dB)
across all 4 UEs this step.
"""
import functools

import numpy as np

# Must match build_four_room_grid.py's SCALE_XY (kept in sync by hand, cross-
# checked against that script's own printed bbox before trusting it). Scales
# every absolute (x,y) position constant below - pickup/dropoff/staging/
# charger locations, room/door boundaries - by the same factor the physical
# scene geometry was scaled by, so "the room is bigger" and "the task field
# is bigger" stay consistent. UE/task DYNAMICS (speed ranges, wait times,
# battery drain rate, box arrival rate, SNR threshold) are deliberately NOT
# scaled - per the user's explicit "keep the UEs and other factors unchanged"
# framing, a bigger floor plan should make the SAME UEs need more ticks to
# cross it and make the SAME 2 TX agents work harder to cover it, not change
# how the UEs themselves behave.
SCALE_XY = 2.0
CACHE_PATH = (f"scene_fixed/tx_radio_map_cache_{SCALE_XY:g}x.npz" if SCALE_XY != 1.0
              else "scene_fixed/tx_radio_map_cache.npz")
NUM_UES = 4
UE_HEIGHT = 1.5
# TX-only step size now (UEs use their own per-trip speed, see BOX_SPEED_RANGE
# below) - a TX steers toward a newly chosen candidate pose at this bounded
# speed rather than teleporting to it.
STEP_SIZE = 0.8
TX_MAX_ANGULAR_STEP_DEG = 20.0  # deg/tick TX antenna can rotate - a separate
# physical quantity from translational step size, bounded so azimuth changes
# gradually too rather than snapping instantly to a new candidate bearing.
NOISE_FLOOR_DBM = -101.0  # thermal noise, 20MHz bandwidth (-174dBm/Hz + 10log10(20e6))

# Valid floor area: union of the 4 room rectangles, connected only through the
# two 1m doors at the row boundary (y in [4.0, 4.05]) - see
# build_four_room_grid.py's DOOR_X_LOCAL for where these come from. Kept for
# other consumers (e.g. build_notebook.py); the task-driven walk below stays
# inside these bounds by construction (every sampled zone/station and every
# routed leg is checked against them), not by rejection sampling.
ROW_BOUNDARY_Y = (4.0 * SCALE_XY, 4.05 * SCALE_XY)
DOOR_X_RANGES = [(2.0 * SCALE_XY, 3.0 * SCALE_XY), (9.05 * SCALE_XY, 10.05 * SCALE_XY)]
DOOR_X_CENTERS = [2.5 * SCALE_XY, 9.55 * SCALE_XY]
FULL_X_RANGE = (-6.0 * SCALE_XY, 18.05 * SCALE_XY)
TOP_Y_RANGE = (4.05 * SCALE_XY, 12.05 * SCALE_XY)
BOTTOM_Y_RANGE = (-4.0 * SCALE_XY, 4.0 * SCALE_XY)
WAYPOINT_TOL = 0.05  # meters; "close enough" to advance to the next waypoint - NOT scaled,
# this is a numerical-arrival tolerance, not a room-proportional distance.

# --- Task-driven UE mobility: box relocation between a random pickup point
# (continuous, left wall) and a random goal point (discrete set, right wall) ---
PICKUP_WALL_X = -5.5 * SCALE_XY   # just inside the left wall of the merged A+B column
DROPOFF_WALL_X = 17.5 * SCALE_XY  # just inside the right wall of the merged B'+A' column
WALL_MARGIN = 0.5 * SCALE_XY      # keep sampled points off the exact room corners
STAGING_POINT = np.array([PICKUP_WALL_X, ROW_BOUNDARY_Y[0]])  # generic idle/home position

# Dropoff: a DISCRETE, non-uniformly-weighted set of goal points - deliberately
# a different KIND of distribution than pickup's continuous draw, not just
# different numbers, so the two sides of the trip are unpredictable in
# different ways.
DROPOFF_GOAL_POINTS = [
    np.array([DROPOFF_WALL_X, -2.5 * SCALE_XY]),  # room A' (bottom-right)
    np.array([DROPOFF_WALL_X, 2.5 * SCALE_XY]),   # room A' (bottom-right)
    np.array([DROPOFF_WALL_X, 6.5 * SCALE_XY]),   # room B' (top-right)
    np.array([DROPOFF_WALL_X, 10.0 * SCALE_XY]),  # room B' (top-right)
]
DROPOFF_GOAL_PROB = [0.15, 0.15, 0.35, 0.35]

# Two box types, each with its OWN speed range (a range, not one fixed value -
# carrying a small box lets a UE move faster, a big box slows it down) and its
# own dropoff wait-time range (heavier boxes take longer to unload).
BOX_TYPES = ("small", "big")
BOX_TYPE_PROB = (0.5, 0.5)
BOX_SPEED_RANGE = {"small": (0.9, 1.3), "big": (0.3, 0.6)}    # m/tick, drawn per trip
BOX_WAIT_RANGE = {"small": (2.0, 6.0), "big": (10.0, 20.0)}   # ticks, drawn per dropoff
EMPTY_SPEED_RANGE = (0.9, 1.3)  # traveling with no box (to pickup, staging, or charging)

# Poisson box-arrival process at the pickup wall.
BOX_ARRIVAL_RATE_PER_TICK = 0.05  # expected new boxes per tick
MAX_QUEUE = 12                     # caps backlog if UEs fall behind arrivals

# --- Route-shape randomization: make coverage genuinely hard to anticipate ---
# A leg between two points in different rows must cross via one of the 2 real
# doors; which door is chosen is now RANDOM (not nearest), and the straight
# line to/through it is perturbed by a Gauss-Markov wander (Liang & Haas 1999
# - blends directed and random motion via one tunable parameter) so a
# coverage agent can't memorize one fixed corridor per leg.
GAUSS_MARKOV_ALPHA = 0.85       # 1=perfectly straight, 0=pure random walk
GAUSS_MARKOV_NOISE_STD_DEG = 35.0  # heading-noise std, degrees, before the alpha blend
MAX_HEADING_DEVIATION_DEG = 45.0   # hard cap on deviation from the CURRENT true
# bearing to target - without this, heading inertia (weighted 85% toward the
# previous tick) plus noise can settle into a stable circling orbit that
# never converges (confirmed empirically: a UE looped a ~9m-radius circle for
# 10,000+ ticks, never approaching its target). Recomputing and clamping
# against the fresh bearing every tick makes that failure mode impossible.

# --- SNR-gated movement: poor coverage physically halts task progress ---
# 0.0dB: raised further from the -4.0dB URLLC-per-transmission-BLER estimate
# (see git history / memory for that derivation) at the user's explicit
# request to make the task harder for a benchmark MAPPO agent - this specific
# value is now a difficulty-tuning choice, not a claimed 3GPP-derived number:
# "signal power equals noise power" is just a clean, easily-interpreted
# reference point once the goal shifted from "physically plausible" to
# "genuinely hard for a baseline RL agent."
MOVEMENT_MIN_SNR_DB = 0.0  # below this a UE halts in place, whatever it's doing.

# --- Battery + charging stations ---
BATTERY_DRAIN_PER_TICK = 0.05  # % per tick -> ~2000 ticks for a full 100%->0% drain
BATTERY_LOW_PCT = 10.0          # triggers a charging trip at/below this level
FAST_CHARGE_RATE_RANGE = (2.0, 4.0)  # %/tick
SLOW_CHARGE_RATE_RANGE = (0.3, 0.8)  # %/tick

# All 8 stations clustered together in ONE corner (bottom-left of room A',
# not spread one-per-room) - deliberately so a coverage agent can learn that
# a UE parked in this specific small area is safe to deprioritize (charging
# itself, and the low-battery travel to reach it, are both already exempt
# from SNR-halting - see _advance_along_route's ignore_snr - so nothing here
# actually NEEDS coverage), freeing it to focus limited coverage on the rest
# of the map where pickup/dropoff/staging traffic actually happens. Placed a
# small fixed distance inside room A's own left/bottom edges (not scaled by
# SCALE_XY - the cluster is meant to stay a small, compact "corner", not grow
# proportionally with the room).
_ROOM_AP_X_LO = 6.05 * SCALE_XY  # left edge of room A' (bottom-right room)
_CHARGE_CLUSTER_ANCHOR = np.array([_ROOM_AP_X_LO + 2.0, BOTTOM_Y_RANGE[0] + 2.0])
_CHARGE_CLUSTER_OFFSETS = [(dx, dy) for dy in (0.0, 1.5) for dx in (0.0, 1.5, 3.0, 4.5)]
FAST_CHARGE_STATIONS = [_CHARGE_CLUSTER_ANCHOR + np.array(off) for off in _CHARGE_CLUSTER_OFFSETS[:4]]
SLOW_CHARGE_STATIONS = [_CHARGE_CLUSTER_ANCHOR + np.array(off) for off in _CHARGE_CLUSTER_OFFSETS[4:]]
ALL_CHARGE_STATIONS = FAST_CHARGE_STATIONS + SLOW_CHARGE_STATIONS
ALL_CHARGE_RATE_RANGES = [FAST_CHARGE_RATE_RANGE] * 4 + [SLOW_CHARGE_RATE_RANGE] * 4


def _sample_pickup_point(rng):
    """Continuous uniform draw anywhere along the left wall, either room."""
    if rng.random() < 0.5:
        y = rng.uniform(BOTTOM_Y_RANGE[0] + WALL_MARGIN, BOTTOM_Y_RANGE[1] - WALL_MARGIN)
    else:
        y = rng.uniform(TOP_Y_RANGE[0] + WALL_MARGIN, TOP_Y_RANGE[1] - WALL_MARGIN)
    return np.array([PICKUP_WALL_X, y])


def _sample_dropoff_point(rng):
    idx = rng.choice(len(DROPOFF_GOAL_POINTS), p=DROPOFF_GOAL_PROB)
    return DROPOFF_GOAL_POINTS[idx].copy()


def _row_of(y):
    return "bottom" if y < ROW_BOUNDARY_Y[0] else "top"


def _route_between(src, dst, rng):
    """Waypoints for one leg from src to dst. Same row (each row is one open
    rectangle, no internal walls) -> direct. Different rows -> via a
    RANDOMLY chosen door (not nearest - see module docstring), 3 legs
    (approach, cross, depart)."""
    src_row, dst_row = _row_of(src[1]), _row_of(dst[1])
    if src_row == dst_row:
        return [np.asarray(dst, dtype=float).copy()]
    door_x = float(rng.choice(DOOR_X_CENTERS))
    if src_row == "bottom":
        approach_y, exit_y = ROW_BOUNDARY_Y[0] - 0.1, ROW_BOUNDARY_Y[1] + 0.1
    else:
        approach_y, exit_y = ROW_BOUNDARY_Y[1] + 0.1, ROW_BOUNDARY_Y[0] - 0.1
    return [np.array([door_x, approach_y]), np.array([door_x, exit_y]),
            np.asarray(dst, dtype=float).copy()]


def _gauss_markov_step(current, target, max_step, heading_deg, rng):
    """One bounded step toward `target`, but the heading is a Gauss-Markov
    blend of the straight-line bearing and the previous tick's heading plus
    fresh noise, rather than a pure straight line (Liang & Haas 1999). Returns
    (new_position, new_heading_deg). Snaps directly to `target` once within
    one step, same as plain waypoint-chasing, so it still reliably arrives.

    The blend is done as a circular (unit-vector) average, NOT a linear
    average of the two angles in degrees - a linear blend of e.g. 350deg and
    10deg (only 20deg apart the short way around) gives ~180deg, the exact
    opposite direction. That wraparound bug was caught empirically: it made
    UEs' headings lock onto essentially random directions and drift forever
    (positions diverging to x~1350, far outside the room) instead of
    converging on their target."""
    delta = target - current
    dist = float(np.linalg.norm(delta))
    if dist <= max_step or dist == 0.0:
        straight_heading = float(np.degrees(np.arctan2(delta[1], delta[0]))) if dist > 0 else heading_deg
        return target.copy(), straight_heading
    straight_heading = float(np.degrees(np.arctan2(delta[1], delta[0])))
    noisy_heading = straight_heading + rng.normal(0.0, GAUSS_MARKOV_NOISE_STD_DEG)
    extra_noise = rng.normal(0.0, (1.0 - GAUSS_MARKOV_ALPHA) * GAUSS_MARKOV_NOISE_STD_DEG)
    prev_rad = np.radians(heading_deg)
    noisy_rad = np.radians(noisy_heading + extra_noise)
    # alpha=1 collapses to pure goal-seeking, alpha=0 to a free (noisy) walk.
    vec = (GAUSS_MARKOV_ALPHA * np.array([np.cos(prev_rad), np.sin(prev_rad)])
           + (1.0 - GAUSS_MARKOV_ALPHA) * np.array([np.cos(noisy_rad), np.sin(noisy_rad)]))
    smoothed = float(np.degrees(np.arctan2(vec[1], vec[0])))
    # Clamp the smoothed heading to within MAX_HEADING_DEVIATION_DEG of the
    # CURRENT tick's true bearing (recomputed fresh above, not carried over) -
    # this is what actually prevents the orbiting failure mode, independent
    # of how "sticky" alpha makes the inertia.
    deviation = (smoothed - straight_heading + 180.0) % 360.0 - 180.0
    deviation = float(np.clip(deviation, -MAX_HEADING_DEVIATION_DEG, MAX_HEADING_DEVIATION_DEG))
    blended = (straight_heading + deviation) % 360.0
    rad = np.radians(blended)
    step = np.array([np.cos(rad), np.sin(rad)]) * max_step
    new_pos = current + step
    # Belt-and-suspenders boundary clamp: noisy wander should never carry a
    # UE through an outer wall even transiently.
    new_pos[0] = np.clip(new_pos[0], FULL_X_RANGE[0], FULL_X_RANGE[1])
    new_pos[1] = np.clip(new_pos[1], BOTTOM_Y_RANGE[0], TOP_Y_RANGE[1])
    return new_pos, blended


def is_valid_floor(x, y):
    if ROW_BOUNDARY_Y[0] <= y <= ROW_BOUNDARY_Y[1]:
        return any(lo <= x <= hi for lo, hi in DOOR_X_RANGES)
    if y > ROW_BOUNDARY_Y[1]:
        return TOP_Y_RANGE[0] <= y <= TOP_Y_RANGE[1] and FULL_X_RANGE[0] <= x <= FULL_X_RANGE[1]
    return BOTTOM_Y_RANGE[0] <= y <= BOTTOM_Y_RANGE[1] and FULL_X_RANGE[0] <= x <= FULL_X_RANGE[1]


TERRITORY_SPLIT_X = 6.025 * SCALE_XY  # midpoint between the two columns' territories


def which_agent_side(x):
    """Agent 1 = rooms A+B (x<=6ish, scaled), agent 2 = rooms B'+A' (x>=6.05ish, scaled),
    matching the AskUserQuestion-confirmed 'restricted to own half' scoping."""
    return 0 if x < TERRITORY_SPLIT_X else 1


def _move_toward_point(current, target, max_step):
    """current, target: (2,) arrays. Moves at most max_step toward target -
    the shared primitive behind both UE random-walk steps and TX steering."""
    delta = target - current
    dist = float(np.linalg.norm(delta))
    if dist <= max_step or dist == 0.0:
        return target.copy()
    return current + delta / dist * max_step


def _move_toward_angle(current_deg, target_deg, max_step_deg):
    """Shortest-path angular step, wraparound-aware (e.g. 350deg -> 10deg is
    a 20deg step, not 340deg)."""
    diff = (target_deg - current_deg + 180.0) % 360.0 - 180.0
    if abs(diff) <= max_step_deg:
        return target_deg % 360.0
    return (current_deg + max_step_deg * np.sign(diff)) % 360.0


def _nearest_index(value, array):
    return int(np.argmin(np.abs(array - value)))


def _nearest_az_index(az_deg, az_array):
    diffs = np.abs((az_array - az_deg + 180.0) % 360.0 - 180.0)
    return int(np.argmin(diffs))


@functools.lru_cache(maxsize=1)
def _load_radio_map_cache(cache_path):
    """The cache is a ~560MB npz (two (nx,ny,n_az,gy,gx) path_gain arrays) - loading it from disk
    is by far the dominant cost of constructing a CoverageMARLEnv. It's a read-only lookup table
    that never changes for the process's lifetime, so cache the loaded arrays here (keyed on path,
    though in practice CACHE_PATH is a fixed constant) and reuse them across every env instance in
    this process instead of re-reading the file each time - discovered as a real bottleneck when
    collect_mappo_offline_dataset.py's per-episode env reconstruction (100x for a full dataset) was
    silently re-reading hundreds of GB from disk."""
    d = np.load(cache_path)
    return dict(agent1_x=d["agent1_x"], agent2_x=d["agent2_x"], agent1_y=d["agent1_y"],
                agent2_y=d["agent2_y"], agent1_path_gain=d["agent1_path_gain"],
                agent2_path_gain=d["agent2_path_gain"], azimuths_deg=d["azimuths_deg"],
                grid_lo=d["grid_lo"], cell_size=d["cell_size"])


class CoverageMARLEnv:
    def __init__(self, tx_power_dbm=0.0, ue_decision_every=1, tx_decision_every=20, seed=None):
        d = _load_radio_map_cache(CACHE_PATH)
        self.agent_x = [d["agent1_x"], d["agent2_x"]]
        self.agent_y = [d["agent1_y"], d["agent2_y"]]
        self.path_gain = [d["agent1_path_gain"], d["agent2_path_gain"]]  # (nx, ny, n_az, gy, gx)
        self.azimuths_deg = d["azimuths_deg"]
        self.grid_lo = d["grid_lo"]
        self.cell_size = float(d["cell_size"])
        self.grid_ny, self.grid_nx = self.path_gain[0].shape[-2:]
        self.tx_power_dbm = tx_power_dbm
        self.tx_power_mw = 10 ** (tx_power_dbm / 10)
        self.ue_decision_every = ue_decision_every
        self.tx_decision_every = tx_decision_every
        self.rng = np.random.default_rng(seed)

        self.t = 0
        self.tx_action = [0, 0]  # flat index into (nx*ny*n_az) candidate space per agent - the target
        self.tx_pos = []          # actual continuous (x, y) per agent - what's physically true this tick
        self.tx_az = []           # actual continuous azimuth (deg) per agent
        self.tx_target_pos = []   # where tx_action says each agent should steer toward
        self.tx_target_az = []
        for i in (0, 1):
            ix, iy, iaz = self._decode_action(i, self.tx_action[i])
            pos = np.array([self.agent_x[i][ix], self.agent_y[i][iy]], dtype=float)
            az = float(self.azimuths_deg[iaz])
            self.tx_pos.append(pos.copy())
            self.tx_az.append(az)
            self.tx_target_pos.append(pos.copy())
            self.tx_target_az.append(az)

        # All 4 UEs start parked at the staging point, empty-handed, waiting
        # for their first box - the natural initial state of a warehouse-style
        # relocation task, at full battery.
        self.ue_pos = np.tile(STAGING_POINT, (NUM_UES, 1)).astype(float)
        self.ue_pos += self.rng.uniform(-0.3, 0.3, size=self.ue_pos.shape)  # avoid exact overlap
        self.ue_state = ["waiting_for_box"] * NUM_UES
        self.ue_route = [[] for _ in range(NUM_UES)]      # remaining waypoints for the current leg
        self.ue_speed = [0.0] * NUM_UES                    # current per-tick step size
        self.ue_heading_deg = [0.0] * NUM_UES               # Gauss-Markov running heading
        self.ue_box_type = [None] * NUM_UES                 # "small"/"big"/None (empty-handed)
        self.ue_wait_timer = [0.0] * NUM_UES                 # ticks remaining while unloading
        self.ue_battery = [100.0] * NUM_UES                  # % of full charge
        self.ue_charge_rate = [0.0] * NUM_UES                 # %/tick, drawn on arrival at a station
        self.ue_charge_station = [None] * NUM_UES              # index into ALL_CHARGE_STATIONS
        self.ue_pre_charge = [None] * NUM_UES                   # snapshot of the interrupted task
        self.box_queue = []                                      # FIFO of box types waiting at pickup
        self.deliveries_completed = 0
        self.charging_trips_started = 0
        # Optional (NUM_UES,) array of live-measured SNR (dB), set by an
        # external driver each tick BEFORE calling step() - lets a real
        # Sionna RT episode script reuse this entire mobility/task/battery
        # state machine unchanged, substituting real ray-traced SNR for the
        # cache lookup. None (default) keeps the existing cache-based path.
        self.live_snr_override = None

    def n_candidates(self, agent_idx):
        nx, ny, naz = self.path_gain[agent_idx].shape[:3]
        return nx * ny * naz

    def _spawn_boxes(self):
        # Unconditional - box arrivals never stop, regardless of any UE being
        # halted (low SNR), charging, or otherwise idle.
        n_new = self.rng.poisson(BOX_ARRIVAL_RATE_PER_TICK)
        for _ in range(n_new):
            if len(self.box_queue) >= MAX_QUEUE:
                break
            box_type = self.rng.choice(BOX_TYPES, p=BOX_TYPE_PROB)
            self.box_queue.append(box_type)

    def _advance_along_route(self, i, ignore_snr=False):
        """Move UE i one tick along its current route using noisy (Gauss-
        Markov) goal-pursuit, unless its instantaneous SNR is below the
        minimum operating threshold - in which case it halts in place this
        tick. `ignore_snr` exempts the emergency low-battery return-to-charge
        leg from this: normal task legs (holding a package, or returning
        empty per the user's own framing) genuinely depend on a live link and
        should halt, but a low-battery return is a safety fallback that must
        not be able to deadlock forever if the UE happens to be stuck exactly
        where coverage is currently poor - without this exemption, an
        untrained/random TX policy reliably strands every UE at 0% battery
        (confirmed empirically: 0 deliveries over 4000 ticks before this fix).
        Returns True once the route is empty (arrived)."""
        if not ignore_snr:
            snr_db = self._ue_snr_db(self.ue_pos[i, 0], self.ue_pos[i, 1], ue_idx=i)
            if snr_db < MOVEMENT_MIN_SNR_DB:
                return False  # halted - poor coverage physically stalls the task
        route = self.ue_route[i]
        if not route:
            return True
        target = route[0]
        new_pos, new_heading = _gauss_markov_step(
            self.ue_pos[i], target, self.ue_speed[i], self.ue_heading_deg[i], self.rng)
        if not is_valid_floor(new_pos[0], new_pos[1]):
            # The noisy wander tried to cut through a solid wall (e.g.
            # overshooting the door slot while crossing rows). A real Sionna
            # RT receiver embedded in wall geometry gives degenerate/-inf
            # SNR, so fall back to a plain straight-line step toward the
            # waypoint for this tick only - the direct line between any two
            # consecutive route waypoints is always valid by construction,
            # since each leg was built to stay within one row's open
            # rectangle or to pass exactly through a door slot.
            new_pos = _move_toward_point(self.ue_pos[i], target, self.ue_speed[i])
            fallback_delta = target - self.ue_pos[i]
            new_heading = float(np.degrees(np.arctan2(fallback_delta[1], fallback_delta[0])))
        self.ue_pos[i] = new_pos
        self.ue_heading_deg[i] = new_heading
        if np.linalg.norm(self.ue_pos[i] - target) <= WAYPOINT_TOL:
            route.pop(0)
        return len(route) == 0

    def _start_charging_trip(self, i):
        """Snapshot whatever task UE i was doing (the box, if any, stays
        conceptually 'with' the UE) and reroute it to a randomly chosen
        charging station."""
        self.ue_pre_charge[i] = dict(
            state=self.ue_state[i], route=list(self.ue_route[i]),
            speed=self.ue_speed[i], wait_timer=self.ue_wait_timer[i],
        )
        station_idx = int(self.rng.integers(len(ALL_CHARGE_STATIONS)))
        self.ue_charge_station[i] = station_idx
        self.ue_charge_rate[i] = float(self.rng.uniform(*ALL_CHARGE_RATE_RANGES[station_idx]))
        self.ue_route[i] = _route_between(self.ue_pos[i], ALL_CHARGE_STATIONS[station_idx], self.rng)
        self.ue_speed[i] = float(self.rng.uniform(*EMPTY_SPEED_RANGE))
        self.ue_state[i] = "to_charging"
        self.charging_trips_started += 1

    def _resume_after_charging(self, i):
        snap = self.ue_pre_charge[i]
        self.ue_state[i] = snap["state"]
        self.ue_route[i] = snap["route"]
        self.ue_speed[i] = snap["speed"]
        self.ue_wait_timer[i] = snap["wait_timer"]
        self.ue_pre_charge[i] = None
        self.ue_charge_station[i] = None

    def _step_ue_tasks(self):
        """Per-UE task state machine: wait for a box at the staging point,
        walk to that box's randomly drawn pickup point, carry it (at a
        box-type-dependent, noisy-heading speed) to a randomly drawn dropoff
        goal point, wait there (box-type-dependent unload time), return to
        staging, and repeat - a continuing task. Battery drain and low-SNR
        halting are checked every tick regardless of which leg a UE is on."""
        self._spawn_boxes()
        for i in range(NUM_UES):
            if self.ue_state[i] != "charging":
                self.ue_battery[i] = max(0.0, self.ue_battery[i] - BATTERY_DRAIN_PER_TICK)
            if (self.ue_battery[i] <= BATTERY_LOW_PCT
                    and self.ue_state[i] not in ("to_charging", "charging")):
                self._start_charging_trip(i)

            state = self.ue_state[i]

            if state == "charging":
                self.ue_battery[i] = min(100.0, self.ue_battery[i] + self.ue_charge_rate[i])
                if self.ue_battery[i] >= 100.0:
                    self._resume_after_charging(i)
                continue  # stationary while charging

            if state == "to_charging":
                if self._advance_along_route(i, ignore_snr=True):
                    self.ue_state[i] = "charging"
                continue

            if state == "waiting_at_dropoff":
                self.ue_wait_timer[i] -= 1.0
                if self.ue_wait_timer[i] <= 0.0:
                    self.deliveries_completed += 1
                    self.ue_box_type[i] = None
                    self.ue_route[i] = _route_between(self.ue_pos[i], STAGING_POINT, self.rng)
                    self.ue_speed[i] = float(self.rng.uniform(*EMPTY_SPEED_RANGE))
                    self.ue_state[i] = "returning_to_staging"
                continue  # stationary while unloading

            if state == "waiting_for_box":
                if self.box_queue:
                    box_type = self.box_queue.pop(0)
                    pickup_pt = _sample_pickup_point(self.rng)
                    self.ue_box_type[i] = box_type
                    self.ue_route[i] = _route_between(self.ue_pos[i], pickup_pt, self.rng)
                    self.ue_speed[i] = float(self.rng.uniform(*EMPTY_SPEED_RANGE))  # fetching, not yet carrying
                    self.ue_state[i] = "to_pickup_point"
                continue  # stationary while no box is available yet

            if state == "to_pickup_point":
                if self._advance_along_route(i):
                    dropoff_pt = _sample_dropoff_point(self.rng)
                    self.ue_route[i] = _route_between(self.ue_pos[i], dropoff_pt, self.rng)
                    self.ue_speed[i] = float(self.rng.uniform(*BOX_SPEED_RANGE[self.ue_box_type[i]]))
                    self.ue_state[i] = "carrying_to_dropoff"
                continue

            if state == "carrying_to_dropoff":
                if self._advance_along_route(i):
                    self.ue_wait_timer[i] = float(self.rng.uniform(*BOX_WAIT_RANGE[self.ue_box_type[i]]))
                    self.ue_state[i] = "waiting_at_dropoff"
                continue

            if state == "returning_to_staging":
                if self._advance_along_route(i):
                    self.ue_state[i] = "waiting_for_box"
                continue

    def _grid_index(self, x, y):
        gx = int(round((x - self.grid_lo[0]) / self.cell_size))
        gy = int(round((y - self.grid_lo[1]) / self.cell_size))
        gx = np.clip(gx, 0, self.grid_nx - 1)
        gy = np.clip(gy, 0, self.grid_ny - 1)
        return gy, gx

    def _decode_action(self, agent_idx, flat_action):
        nx, ny, naz = self.path_gain[agent_idx].shape[:3]
        ix, rem = divmod(flat_action, ny * naz)
        iy, iaz = divmod(rem, naz)
        return ix, iy, iaz

    def _received_power_mw(self, x, y):
        """Sum both agents' contribution (linear power) at a UE's position,
        using each agent's actual current continuous pose (not its target) -
        snapped to the nearest precomputed candidate cell for the lookup."""
        gy, gx = self._grid_index(x, y)
        total_mw = 0.0
        for agent_idx in (0, 1):
            ix = _nearest_index(self.tx_pos[agent_idx][0], self.agent_x[agent_idx])
            iy = _nearest_index(self.tx_pos[agent_idx][1], self.agent_y[agent_idx])
            iaz = _nearest_az_index(self.tx_az[agent_idx], self.azimuths_deg)
            pg = self.path_gain[agent_idx][ix, iy, iaz, gy, gx]
            total_mw += pg * self.tx_power_mw
        return total_mw

    def _ue_snr_db(self, x, y, ue_idx=None):
        if ue_idx is not None and self.live_snr_override is not None:
            return float(self.live_snr_override[ue_idx])
        rx_mw = self._received_power_mw(x, y)
        noise_mw = 10 ** (NOISE_FLOOR_DBM / 10)
        return 10 * np.log10(rx_mw / noise_mw) if rx_mw > 0 else -200.0

    @staticmethod
    def _carrying_code(box_type):
        """0 = empty-handed, 1 = carrying a small box, 2 = carrying a big one -
        the discrete analog of Xi_t/A_t-style state DMWM feeds its world
        model, here standing in for "task state" rather than packet CAoI."""
        return 0 if box_type is None else (1 if box_type == "small" else 2)

    def step(self, actions=None):
        """actions: optional (agent1_flat_action, agent2_flat_action) - a new
        candidate TARGET pose, applied only every `tx_decision_every` steps
        (coverage decisions happen slowly). Every tick (not just decision
        ticks), each TX agent physically steers toward its current target at
        a bounded speed (STEP_SIZE m/tick translation, TX_MAX_ANGULAR_STEP_DEG
        deg/tick rotation) - it does not teleport to the target. Returns
        (observations, reward, info) - no `done`, this is a continuing task."""
        if actions is not None and self.t % self.tx_decision_every == 0:
            self.tx_action = list(actions)
            for i in (0, 1):
                ix, iy, iaz = self._decode_action(i, self.tx_action[i])
                self.tx_target_pos[i] = np.array([self.agent_x[i][ix], self.agent_y[i][iy]], dtype=float)
                self.tx_target_az[i] = float(self.azimuths_deg[iaz])

        for i in (0, 1):
            self.tx_pos[i] = _move_toward_point(self.tx_pos[i], self.tx_target_pos[i], STEP_SIZE)
            self.tx_az[i] = _move_toward_angle(self.tx_az[i], self.tx_target_az[i], TX_MAX_ANGULAR_STEP_DEG)

        if self.t % self.ue_decision_every == 0:
            self._step_ue_tasks()

        snr_db_per_ue = np.array([self._ue_snr_db(self.ue_pos[i, 0], self.ue_pos[i, 1], ue_idx=i)
                                   for i in range(NUM_UES)])

        reward = float(np.mean(snr_db_per_ue))
        obs = self._get_observations()
        halted = [self.ue_state[i] in ("to_pickup_point", "carrying_to_dropoff", "returning_to_staging")
                  and snr_db_per_ue[i] < MOVEMENT_MIN_SNR_DB for i in range(NUM_UES)]
        info = {"snr_db_per_ue": snr_db_per_ue, "ue_pos": self.ue_pos.copy(),
                "tx_action": list(self.tx_action),
                "tx_pose": [(float(self.tx_pos[i][0]), float(self.tx_pos[i][1]), float(self.tx_az[i]))
                            for i in (0, 1)],
                "ue_state": list(self.ue_state), "ue_box_type": list(self.ue_box_type),
                "ue_speed": list(self.ue_speed), "ue_battery": list(self.ue_battery),
                "ue_halted_low_snr": halted, "ue_charge_station": list(self.ue_charge_station),
                "box_queue_len": len(self.box_queue), "deliveries_completed": self.deliveries_completed,
                "charging_trips_started": self.charging_trips_started}
        self.t += 1
        return obs, reward, info

    def _get_observations(self):
        """Partial observability: each agent only sees UEs currently on its
        own side (its own two rooms), plus its own actual current TX pose
        (not the target it's steering toward - what it commanded is already
        known to it trivially; what matters as an observation is where it
        physically is right now, since with bounded speed that can lag the
        target). Each visible UE now carries the same *kind* of input DMWM's
        world model gets from Sionna RT: a channel-quality feature (here, SNR
        at the UE's actual position, the cache's equivalent of a ray-traced
        link feature) alongside its location, a task-state feature (carrying
        code) standing in for DMWM's packet-completeness state, and battery
        level - so the learner can see when a UE is headed for/parked at a
        charging station and doesn't need active coverage there."""
        obs = [[], []]
        for i in range(NUM_UES):
            side = which_agent_side(self.ue_pos[i, 0])
            snr_db = self._ue_snr_db(self.ue_pos[i, 0], self.ue_pos[i, 1], ue_idx=i)
            carrying = self._carrying_code(self.ue_box_type[i])
            obs[side].append((self.ue_pos[i].copy(), snr_db, carrying, self.ue_battery[i]))
        return {
            "agent1": {"visible_ue": obs[0],
                       "own_tx_pose": (float(self.tx_pos[0][0]), float(self.tx_pos[0][1]), float(self.tx_az[0]))},
            "agent2": {"visible_ue": obs[1],
                       "own_tx_pose": (float(self.tx_pos[1][0]), float(self.tx_pos[1][1]), float(self.tx_az[1]))},
        }


if __name__ == "__main__":
    env = CoverageMARLEnv(tx_power_dbm=0.0, seed=0)
    rewards = []
    halted_ticks = 0
    for t in range(4000):
        actions = None
        if t % env.tx_decision_every == 0:
            actions = (env.rng.integers(env.n_candidates(0)), env.rng.integers(env.n_candidates(1)))
        obs, reward, info = env.step(actions)
        rewards.append(reward)
        halted_ticks += sum(info["ue_halted_low_snr"])
    print(f"Smoke test: {len(rewards)} steps, mean reward (avg SNR dB) = {np.mean(rewards):.2f}, "
          f"min={np.min(rewards):.2f}, max={np.max(rewards):.2f}")
    print(f"deliveries completed: {info['deliveries_completed']} over {len(rewards)} ticks, "
          f"final box_queue_len={info['box_queue_len']}")
    print(f"charging trips started: {info['charging_trips_started']}, "
          f"UE-ticks halted on low SNR: {halted_ticks}")
    print(f"final ue_state={info['ue_state']}, ue_box_type={info['ue_box_type']}, "
          f"ue_battery={[f'{b:.1f}' for b in info['ue_battery']]}")
    print(f"agent1 sees {len(obs['agent1']['visible_ue'])} UEs, "
          f"agent2 sees {len(obs['agent2']['visible_ue'])} UEs (last step)")
