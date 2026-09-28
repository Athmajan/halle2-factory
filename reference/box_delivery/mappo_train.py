"""MAPPO training loop for the 2 coverage-TX-agent Dec-POMDP
(CoverageMARLEnv, wrapped by MacroStepCoverageEnv).

CTDE, matching arXiv:2607.28002: one centralized critic V(global_state)
trained on the shared team reward (mean SNR in dB across all 4 UEs), one
actor with parameters shared across both agents (valid since local obs are
normalized into each agent's own territory-relative coordinates - see
mappo_env_wrapper.py). Continuing task: the environment never resets (UEs
random-walk indefinitely, no episode boundary) - rollouts are truncated
windows with a bootstrapped value at the end, standard for continuing MDPs.

Usage:
    python mappo_train.py --iterations 200 --rollout_len 64
Writes checkpoints/mappo_{actor,critic}.pt and logs/mappo_train_log.csv.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

from mappo_env_wrapper import MacroStepCoverageEnv, LOCAL_OBS_DIM, GLOBAL_STATE_DIM
from mappo_networks import SharedActor, CentralizedCritic


def compute_gae(rewards, values, next_value, gamma, lam):
    """rewards, values: (T,) numpy. next_value: scalar (bootstrap). No dones -
    continuing task, every step is treated as non-terminal."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        v_next = values[t + 1] if t + 1 < T else next_value
        delta = rewards[t] + gamma * v_next - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--rollout_len", type=int, default=64)
    p.add_argument("--tx_decision_every", type=int, default=20)
    p.add_argument("--tx_power_dbm", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=1e-3,
                    help="higher than actor_lr by default - value_loss was noisy in the first run")
    p.add_argument("--entropy_coef_start", type=float, default=0.02)
    p.add_argument("--entropy_coef_end", type=float, default=0.003,
                    help="linearly annealed over --iterations so exploration doesn't collapse too early")
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--value_clip_eps", type=float, default=20.0,
                    help="PPO2-style value clipping, in scaled-reward units; with gamma=0.99 and "
                         "reward_scale=0.05, returns run O(25-150), so this must be large relative "
                         "to a single reward step - too small (e.g. 0.2) saturates the clamp on "
                         "nearly every sample and kills the critic's gradient almost entirely")
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward_scale", type=float, default=0.05,
                    help="rewards are avg-SNR in dB (can be tens); scale down for stable value targets")
    p.add_argument("--min_snr_db", type=float, default=None,
                    help="defaults to mappo_env_wrapper.MIN_SNR_DB (-6.9dB, this repo's CQI-1 threshold)")
    p.add_argument("--constraint_lambda", type=float, default=1.0,
                    help="weight on the mean per-UE SNR shortfall below min_snr_db, added to the reward")
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--ckpt_every", type=int, default=20)
    p.add_argument("--out_dir", type=str, default=".")
    p.add_argument("--resume", type=str, default=None,
                    help="path prefix to load {actor,critic}.pt from before training (e.g. ./checkpoints/mappo)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    min_snr_db = args.min_snr_db if args.min_snr_db is not None else None
    env_kwargs = dict(tx_power_dbm=args.tx_power_dbm, tx_decision_every=args.tx_decision_every,
                       seed=args.seed, constraint_lambda=args.constraint_lambda)
    if min_snr_db is not None:
        env_kwargs["min_snr_db"] = min_snr_db
    env = MacroStepCoverageEnv(**env_kwargs)
    head_sizes = env.head_sizes
    print(f"min_snr_db={env.min_snr_db}, constraint_lambda={env.constraint_lambda}")

    actor = SharedActor(LOCAL_OBS_DIM, head_sizes).to(device)
    critic = CentralizedCritic(GLOBAL_STATE_DIM).to(device)
    if args.resume:
        actor.load_state_dict(torch.load(f"{args.resume}_actor.pt", map_location=device))
        critic.load_state_dict(torch.load(f"{args.resume}_critic.pt", map_location=device))
        print(f"Resumed actor/critic from {args.resume}_{{actor,critic}}.pt")
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "mappo_train_log.csv")
    log_f = open(log_path, "w", newline="")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["iteration", "mean_raw_reward", "mean_shaped_reward", "mean_violation_db",
                          "actor_loss", "value_loss", "entropy", "entropy_coef", "approx_kl", "elapsed_s"])

    local_obs, glob = env.reset()
    t_start = time.time()

    for it in range(1, args.iterations + 1):
        buf_local = [[], []]      # per agent: list of (LOCAL_OBS_DIM,)
        buf_action = [[], []]     # per agent: list of (3,)
        buf_logprob = [[], []]    # per agent: list of scalar
        buf_global = []
        buf_value = []
        buf_reward = []
        buf_raw_reward = []
        buf_violation = []

        for step in range(args.rollout_len):
            local_t = [torch.as_tensor(local_obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                       for i in (0, 1)]
            global_t = torch.as_tensor(glob, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                actions, logprobs, values = [], [], critic(global_t).item()
                for i in (0, 1):
                    a, lp, _ = actor.act(local_t[i])
                    actions.append(a.squeeze(0).cpu().numpy())
                    logprobs.append(lp.item())

            action_tuples = [tuple(int(x) for x in actions[i]) for i in (0, 1)]
            next_local, next_glob, reward, info = env.macro_step(action_tuples)

            for i in (0, 1):
                buf_local[i].append(local_obs[i])
                buf_action[i].append(actions[i])
                buf_logprob[i].append(logprobs[i])
            buf_global.append(glob)
            buf_value.append(values)
            buf_reward.append(reward * args.reward_scale)
            buf_raw_reward.append(info["raw_reward_mean"])
            buf_violation.append(info["constraint_violation_mean"])

            local_obs, glob = next_local, next_glob

        with torch.no_grad():
            next_value = critic(torch.as_tensor(glob, dtype=torch.float32,
                                                 device=device).unsqueeze(0)).item()

        rewards_arr = np.array(buf_reward, dtype=np.float32)
        values_arr = np.array(buf_value, dtype=np.float32)
        advantages, returns = compute_gae(rewards_arr, values_arr, next_value, args.gamma, args.lam)
        advantages_n = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Both agents share the shared team-reward advantage/return; stack their
        # (obs, action, logprob) into one batch of size 2*rollout_len for the
        # parameter-shared actor update. Global-state/value data is per-timestep
        # (not doubled) for the critic update.
        obs_batch = torch.as_tensor(np.concatenate([buf_local[0], buf_local[1]], axis=0),
                                     dtype=torch.float32, device=device)
        action_batch = torch.as_tensor(np.concatenate([buf_action[0], buf_action[1]], axis=0),
                                        dtype=torch.long, device=device)
        old_logprob_batch = torch.as_tensor(np.concatenate([buf_logprob[0], buf_logprob[1]], axis=0),
                                             dtype=torch.float32, device=device)
        adv_batch = torch.as_tensor(np.concatenate([advantages_n, advantages_n], axis=0),
                                     dtype=torch.float32, device=device)

        global_batch = torch.as_tensor(np.array(buf_global), dtype=torch.float32, device=device)
        return_batch = torch.as_tensor(returns, dtype=torch.float32, device=device)

        old_value_batch = torch.as_tensor(values_arr, dtype=torch.float32, device=device)

        n_actor = obs_batch.shape[0]
        n_critic = global_batch.shape[0]
        actor_idx = np.arange(n_actor)
        critic_idx = np.arange(n_critic)
        mb_actor = max(1, n_actor // args.minibatches)
        mb_critic = max(1, n_critic // args.minibatches)

        frac = (it - 1) / max(1, args.iterations - 1)
        entropy_coef = args.entropy_coef_start + frac * (args.entropy_coef_end - args.entropy_coef_start)

        last_actor_loss = last_value_loss = last_entropy = last_kl = 0.0
        for epoch in range(args.epochs):
            np.random.shuffle(actor_idx)
            np.random.shuffle(critic_idx)
            for start in range(0, n_actor, mb_actor):
                mb = actor_idx[start:start + mb_actor]
                logprob, entropy = actor.evaluate_actions(obs_batch[mb], action_batch[mb])
                ratio = torch.exp(logprob - old_logprob_batch[mb])
                surr1 = ratio * adv_batch[mb]
                surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv_batch[mb]
                actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy.mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                actor_opt.step()

                with torch.no_grad():
                    last_kl = (old_logprob_batch[mb] - logprob).mean().item()
                last_actor_loss = actor_loss.item()
                last_entropy = entropy.mean().item()

            for start in range(0, n_critic, mb_critic):
                mb = critic_idx[start:start + mb_critic]
                value_pred = critic(global_batch[mb])
                # PPO2-style value clipping: damps how far the critic can move
                # from its rollout-time estimate in one update, same spirit as
                # the policy ratio clip - used because value_loss was noisy
                # (occasional 1-2 order-of-magnitude spikes) in the first run.
                value_clipped = old_value_batch[mb] + torch.clamp(
                    value_pred - old_value_batch[mb], -args.value_clip_eps, args.value_clip_eps)
                loss_unclipped = (value_pred - return_batch[mb]) ** 2
                loss_clipped = (value_clipped - return_batch[mb]) ** 2
                value_loss = args.value_coef * torch.max(loss_unclipped, loss_clipped).mean()

                critic_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                critic_opt.step()
                last_value_loss = value_loss.item()

        mean_raw_reward = float(np.mean(buf_raw_reward))
        mean_shaped_reward = rewards_arr.mean() / args.reward_scale
        mean_violation = float(np.mean(buf_violation))
        elapsed = time.time() - t_start
        if it % args.log_every == 0:
            print(f"iter {it:4d}/{args.iterations} | raw(dB)={mean_raw_reward:7.2f} | "
                  f"shaped={mean_shaped_reward:7.2f} | violation(dB)={mean_violation:6.2f} | "
                  f"actor_loss={last_actor_loss:+.4f} | value_loss={last_value_loss:.4f} | "
                  f"entropy={last_entropy:.3f} (coef={entropy_coef:.4f}) | approx_kl={last_kl:+.4f} | "
                  f"{elapsed:6.1f}s")
        log_writer.writerow([it, mean_raw_reward, mean_shaped_reward, mean_violation, last_actor_loss,
                              last_value_loss, last_entropy, entropy_coef, last_kl, elapsed])
        log_f.flush()

        if it % args.ckpt_every == 0 or it == args.iterations:
            torch.save(actor.state_dict(), os.path.join(ckpt_dir, "mappo_actor.pt"))
            torch.save(critic.state_dict(), os.path.join(ckpt_dir, "mappo_critic.pt"))

    log_f.close()
    print(f"Done. Checkpoints in {ckpt_dir}/, log at {log_path}")


if __name__ == "__main__":
    main()
