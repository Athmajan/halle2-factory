"""Macro-step wrapper around CoverageMARLEnv for MAPPO training.

CoverageMARLEnv ticks every simulated timestep (UEs random-walk every tick;
TX agents only re-decide every `tx_decision_every` ticks). For RL training we
treat one full tx-decision interval as a single "macro-step": the agents pick
a new (ix, iy, iaz) candidate pose, the environment ticks forward
`tx_decision_every` times under that fixed pose while UEs keep moving, and the
macro-step's reward is the mean of the per-tick shared rewards (average SNR in
dB) over the interval. This also produces fixed-size, network-friendly
observations out of the environment's variable-length visible-UE lists.

Two observation types, matching MAPPO's CTDE split:
  - local obs (per agent, what the decentralized actor sees): own current TX
    pose + up to 4 own-side UE slots (visibility mask + normalized position),
    sorted by (y, x) for a deterministic slot assignment since the env does
    not expose UE identity in its partial observations.
  - global state (centralized critic only): true position of all 4 UEs
    (regardless of visibility) + both agents' TX poses.
All positions are normalized to each agent's own candidate-grid bounds (local
obs) or the full scene bounds (global state UE positions) into [0, 1].
"""
import numpy as np

from coverage_marl_env import (
    CoverageMARLEnv, NUM_UES, FULL_X_RANGE, BOTTOM_Y_RANGE, TOP_Y_RANGE, MOVEMENT_MIN_SNR_DB,
)

# own tx pose (4) + 4 UE slots x (visible, x_norm, y_norm, snr_norm, carrying_norm, battery_norm)
LOCAL_OBS_DIM = 4 + 4 * 6
# 4 UE x (x,y,snr_norm,carrying_norm,battery_norm) + 2 agents x (x,y,sin_az,cos_az)
GLOBAL_STATE_DIM = NUM_UES * 5 + 2 * 4

# Same constant CoverageMARLEnv uses to halt UE movement (imported, not
# duplicated, so the reward-shaping constraint and the actual halting
# behavior can never drift apart): -4.0dB, a URLLC/factory-automation-class
# threshold, not the eMBB CQI-1 value (-6.9dB) this project uses elsewhere -
# see coverage_marl_env.py's MOVEMENT_MIN_SNR_DB comment for the reasoning.
MIN_SNR_DB = MOVEMENT_MIN_SNR_DB

# Generous clamp range for normalizing per-UE SNR into [0,1] as a network
# input - wide enough to cover both the near-TX ceiling and the env's -200dB
# zero-gain-cell fallback without the clip collapsing all "bad" cells to the
# same value at the low end (that fallback is a known cache artifact, not a
# real SNR, see wisegrt-eval/README.md's "artifact the movie itself surfaced").
SNR_NORM_RANGE = (-40.0, 50.0)


def _norm(val, lo, hi):
    if hi <= lo:
        return 0.0
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))


class MacroStepCoverageEnv:
    def __init__(self, tx_power_dbm=0.0, tx_decision_every=20, seed=None, reward_clip=(-60.0, 40.0),
                 min_snr_db=MIN_SNR_DB, constraint_lambda=1.0):
        self.env = CoverageMARLEnv(tx_power_dbm=tx_power_dbm, ue_decision_every=1,
                                    tx_decision_every=tx_decision_every, seed=seed)
        self.tx_decision_every = tx_decision_every
        self.reward_clip = reward_clip
        self.min_snr_db = min_snr_db
        self.constraint_lambda = constraint_lambda
        self.n_candidates = [self.env.n_candidates(0), self.env.n_candidates(1)]
        self.head_sizes = self._head_sizes()
        self._last_obs_raw = self.env._get_observations()

    def _head_sizes(self):
        nx0, ny0, naz0 = self.env.path_gain[0].shape[:3]
        nx1, ny1, naz1 = self.env.path_gain[1].shape[:3]
        assert (nx0, ny0, naz0) == (nx1, ny1, naz1), "networks assume matching grid shapes across agents"
        return nx0, ny0, naz0

    def encode_action(self, agent_idx, ix, iy, iaz):
        nx, ny, naz = self.head_sizes
        return int(ix) * (ny * naz) + int(iy) * naz + int(iaz)

    def _own_tx_pose_norm(self, agent_idx, pose):
        """pose: (x, y, az_deg) - the agent's actual current physical pose
        (env now steers TXs toward a target at bounded speed rather than
        teleporting, so this is not always exactly the last chosen action)."""
        x, y, az_deg = pose
        x_lo, x_hi = self.env.agent_x[agent_idx].min(), self.env.agent_x[agent_idx].max()
        y_lo, y_hi = self.env.agent_y[agent_idx].min(), self.env.agent_y[agent_idx].max()
        x_n = _norm(x, x_lo, x_hi)
        y_n = _norm(y, y_lo, y_hi)
        az_rad = np.deg2rad(az_deg)
        return [x_n, y_n, float(np.sin(az_rad)), float(np.cos(az_rad))]

    def _local_obs(self, agent_idx, obs_raw):
        key = f"agent{agent_idx + 1}"
        own_tx = self._own_tx_pose_norm(agent_idx, obs_raw[key]["own_tx_pose"])
        ues = obs_raw[key]["visible_ue"]  # list of ((x,y), snr_db, carrying_code, battery_pct)
        x_lo, x_hi = self.env.agent_x[agent_idx].min(), self.env.agent_x[agent_idx].max()
        y_lo, y_hi = self.env.agent_y[agent_idx].min(), self.env.agent_y[agent_idx].max()
        ues_sorted = sorted(ues, key=lambda u: (u[0][1], u[0][0]))
        slots = []
        for i in range(4):
            if i < len(ues_sorted):
                (x, y), snr_db, carrying, battery = ues_sorted[i]
                slots.extend([1.0, _norm(x, x_lo, x_hi), _norm(y, y_lo, y_hi),
                              _norm(snr_db, *SNR_NORM_RANGE), carrying / 2.0, battery / 100.0])
            else:
                slots.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return np.array(own_tx + slots, dtype=np.float32)

    def _global_state(self, obs_raw, ue_pos):
        x_lo, x_hi = FULL_X_RANGE
        y_lo, y_hi = BOTTOM_Y_RANGE[0], TOP_Y_RANGE[1]
        ue_part = []
        for i in range(NUM_UES):
            x, y = ue_pos[i]
            snr_db = self.env._ue_snr_db(x, y)
            carrying = self.env._carrying_code(self.env.ue_box_type[i])
            battery = self.env.ue_battery[i]
            ue_part.extend([_norm(x, x_lo, x_hi), _norm(y, y_lo, y_hi),
                            _norm(snr_db, *SNR_NORM_RANGE), carrying / 2.0, battery / 100.0])
        tx_part = []
        for agent_idx in (0, 1):
            key = f"agent{agent_idx + 1}"
            tx_part.extend(self._own_tx_pose_norm(agent_idx, obs_raw[key]["own_tx_pose"]))
        return np.array(ue_part + tx_part, dtype=np.float32)

    def reset(self):
        self.env = CoverageMARLEnv(tx_power_dbm=self.env.tx_power_dbm, ue_decision_every=1,
                                    tx_decision_every=self.tx_decision_every, seed=None)
        self._last_obs_raw = self.env._get_observations()
        local = [self._local_obs(0, self._last_obs_raw), self._local_obs(1, self._last_obs_raw)]
        glob = self._global_state(self._last_obs_raw, self.env.ue_pos)
        return local, glob

    def macro_step(self, action_ix_iy_iaz):
        """action_ix_iy_iaz: [(ix1,iy1,iaz1), (ix2,iy2,iaz2)]. Ticks the env
        forward `tx_decision_every` steps, applying the new action on the
        first tick.

        Reward is a soft-constrained objective: mean SNR (dB, the coverage/
        throughput proxy) minus `constraint_lambda` times the mean per-UE,
        per-tick shortfall below `min_snr_db` (a hinge penalty - 0 when a UE
        is above the floor, growing linearly with how far below it is). This
        is the user's original framing ("optimal coverage... respecting a
        minimum SNR level, and maximize throughput") as a single scalar
        objective for unconstrained PPO, rather than a hard constraint.

        Returns (local_obs[2], global_state, shaped_reward, info) - info
        carries the unconstrained raw mean SNR and the violation stats
        separately so training can be monitored on both axes."""
        flat_actions = [self.encode_action(i, *action_ix_iy_iaz[i]) for i in (0, 1)]
        raw_rewards = []
        violations = []
        obs_raw = None
        for tick in range(self.tx_decision_every):
            actions = flat_actions if tick == 0 else None
            obs_raw, reward, info = self.env.step(actions)
            per_ue = np.asarray(info["snr_db_per_ue"], dtype=np.float32)
            if self.reward_clip:
                per_ue = np.clip(per_ue, *self.reward_clip)
            raw_rewards.append(float(per_ue.mean()))
            violations.append(float(np.clip(self.min_snr_db - per_ue, 0.0, None).mean()))
        self._last_obs_raw = obs_raw
        local = [self._local_obs(0, obs_raw), self._local_obs(1, obs_raw)]
        glob = self._global_state(obs_raw, self.env.ue_pos)
        mean_raw_reward = float(np.mean(raw_rewards))
        mean_violation = float(np.mean(violations))
        shaped_reward = mean_raw_reward - self.constraint_lambda * mean_violation
        info = {"raw_reward_mean": mean_raw_reward, "constraint_violation_mean": mean_violation,
                "shaped_reward": shaped_reward, "snr_db_per_ue_last_tick": info["snr_db_per_ue"]}
        return local, glob, shaped_reward, info
