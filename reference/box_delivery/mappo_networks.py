"""Shared-parameter decentralized actor + centralized critic for MAPPO.

CTDE (Perner et al., arXiv:2607.28002): one centralized critic V(global_state)
trained on the shared team reward (both coverage agents optimize the same
average-SNR objective, so a single value function is correct - there is no
need for per-agent critics here), and one actor network whose parameters are
shared across both TX agents (they are structurally symmetric - each covers
two mirrored rooms - so parameter sharing is valid once local observations
are normalized into each agent's own territory-relative coordinates, which
mappo_env_wrapper.py already does).

The action space per agent is factored multi-discrete (ix, iy, iaz) rather
than one flat 1320-way categorical - this mirrors precompute_tx_radio_map_cache
.py's (x, y, azimuth) candidate grid directly and is far more sample-efficient
than a flat head over 1320 classes.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical


class SharedActor(nn.Module):
    def __init__(self, obs_dim, head_sizes, hidden=128):
        super().__init__()
        nx, ny, naz = head_sizes
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.head_ix = nn.Linear(hidden, nx)
        self.head_iy = nn.Linear(hidden, ny)
        self.head_iaz = nn.Linear(hidden, naz)

    def forward(self, obs):
        h = self.trunk(obs)
        return self.head_ix(h), self.head_iy(h), self.head_iaz(h)

    def act(self, obs, deterministic=False):
        """obs: (B, obs_dim). Returns action (B,3) long, logprob (B,), entropy (B,)."""
        lix, liy, liaz = self.forward(obs)
        dists = [Categorical(logits=l) for l in (lix, liy, liaz)]
        if deterministic:
            acts = [d.probs.argmax(dim=-1) for d in dists]
        else:
            acts = [d.sample() for d in dists]
        logprob = sum(d.log_prob(a) for d, a in zip(dists, acts))
        entropy = sum(d.entropy() for d in dists)
        action = torch.stack(acts, dim=-1)
        return action, logprob, entropy

    def evaluate_actions(self, obs, action):
        """action: (B,3) long. Returns logprob (B,), entropy (B,)."""
        lix, liy, liaz = self.forward(obs)
        dists = [Categorical(logits=l) for l in (lix, liy, liaz)]
        logprob = sum(d.log_prob(action[:, i]) for i, d in enumerate(dists))
        entropy = sum(d.entropy() for d in dists)
        return logprob, entropy


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)
