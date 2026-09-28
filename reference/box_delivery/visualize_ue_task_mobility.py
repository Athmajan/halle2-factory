"""Preview movie of the task-driven UE (Rx) box-relocation mobility model,
with NO trained policy involved - TX agents stay at their fixed initial pose
(actions=None every tick) so the movie isolates UE movement/behavior before
any MAPPO training happens. Pure matplotlib animation driven directly by
CoverageMARLEnv (cache-only, no Sionna scene load).

Usage: python visualize_ue_task_mobility.py --ticks 1500 --fps 20
Writes renders/ue_task_mobility_preview.mp4 (via imageio's bundled ffmpeg).
"""
import argparse

import imageio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from coverage_marl_env import (
    CoverageMARLEnv, PICKUP_WALL_X, DROPOFF_WALL_X, DROPOFF_GOAL_POINTS, DROPOFF_GOAL_PROB,
    STAGING_POINT, FAST_CHARGE_STATIONS, SLOW_CHARGE_STATIONS, BOTTOM_Y_RANGE, TOP_Y_RANGE,
    WALL_MARGIN, DOOR_X_RANGES, FULL_X_RANGE, SCALE_XY,
)

# Schematic room rectangles (x_lo, x_hi, y_lo, y_hi), derived from the SAME
# SCALE_XY as the rest of the geometry (imported, not re-hardcoded).
ROOMS = [
    ("A", -6.0 * SCALE_XY, 6.0 * SCALE_XY, 4.05 * SCALE_XY, 12.05 * SCALE_XY),
    ("B", -6.0 * SCALE_XY, 6.0 * SCALE_XY, -4.0 * SCALE_XY, 4.0 * SCALE_XY),
    ("B'", 6.05 * SCALE_XY, 18.05 * SCALE_XY, 4.05 * SCALE_XY, 12.05 * SCALE_XY),
    ("A'", 6.05 * SCALE_XY, 18.05 * SCALE_XY, -4.0 * SCALE_XY, 4.0 * SCALE_XY),
]

UE_COLORS = ["tab:red", "tab:orange", "tab:green", "tab:purple"]
TRAIL_LEN = 50

# Marker style keyed by carrying state: empty-handed vs. carrying small/big.
STATE_STYLE = {
    None: dict(marker="o"),
    "small": dict(marker="s"),
    "big": dict(marker="D"),
}
MOVING_STATES = ("to_pickup_point", "carrying_to_dropoff", "returning_to_staging", "to_charging")


def rollout(ticks, seed):
    env = CoverageMARLEnv(seed=seed)
    frames = []
    for tick in range(ticks):
        # No trained policy: TX agents take a fresh RANDOM action every
        # decision window, same as real pre-training exploration (and the
        # env's own smoke test) - a single fixed pose for the whole preview
        # essentially never covers the full 24x16m, 4-room map (confirmed
        # empirically: 94-98% of UE-ticks halted across 3 seeds), which is
        # the coverage problem working as intended but makes an uninformative
        # frozen movie. Roaming random coverage lets different areas get
        # served at different times, matching the smoke test's healthy
        # ~40% halted / real delivery throughput.
        actions = (env.rng.integers(env.n_candidates(0)), env.rng.integers(env.n_candidates(1))) \
            if tick % env.tx_decision_every == 0 else None
        _obs, reward, info = env.step(actions)
        frames.append(dict(
            tick=tick,
            ue_pos=env.ue_pos.copy(),
            ue_state=list(info["ue_state"]),
            ue_box_type=list(info["ue_box_type"]),
            ue_speed=list(info["ue_speed"]),
            ue_battery=list(info["ue_battery"]),
            ue_halted=list(info["ue_halted_low_snr"]),
            snr_per_ue=np.asarray(info["snr_db_per_ue"], dtype=np.float32).copy(),
            box_queue_len=info["box_queue_len"],
            deliveries_completed=info["deliveries_completed"],
            charging_trips_started=info["charging_trips_started"],
        ))
    return frames


def render_movie(frames, out_path, fps):
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=100)
    margin = 2.0 * SCALE_XY
    ax.set_xlim(FULL_X_RANGE[0] - margin, FULL_X_RANGE[1] + margin)
    ax.set_ylim(BOTTOM_Y_RANGE[0] - margin, TOP_Y_RANGE[1] + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Task-driven UE mobility preview (no policy - TX poses fixed)")

    for label, x_lo, x_hi, y_lo, y_hi in ROOMS:
        ax.add_patch(Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                                fill=False, edgecolor="gray", linewidth=1.5))
        ax.text(x_lo + 0.3, y_hi - 0.6, label, fontsize=11, color="gray", fontweight="bold")
    for x0, x1 in DOOR_X_RANGES:
        ax.plot([x0, x1], [4.025, 4.025], color="white", linewidth=4, zorder=2)
        ax.plot([x0, x1], [4.025, 4.025], color="gray", linewidth=1, linestyle=":", zorder=3)

    # Pickup band (continuous distribution, left wall) and dropoff goal points
    # (discrete weighted distribution, right wall).
    ax.plot([PICKUP_WALL_X, PICKUP_WALL_X],
            [BOTTOM_Y_RANGE[0] + WALL_MARGIN, BOTTOM_Y_RANGE[1] - WALL_MARGIN],
            color="gold", linewidth=5, alpha=0.6, zorder=1, label="pickup band (uniform)")
    ax.plot([PICKUP_WALL_X, PICKUP_WALL_X],
            [TOP_Y_RANGE[0] + WALL_MARGIN, TOP_Y_RANGE[1] - WALL_MARGIN],
            color="gold", linewidth=5, alpha=0.6, zorder=1)
    for pt, prob in zip(DROPOFF_GOAL_POINTS, DROPOFF_GOAL_PROB):
        ax.scatter(*pt, marker="X", s=150 + 700 * prob, c="black", zorder=4)
    ax.scatter([], [], marker="X", c="black", s=150, label="dropoff goal points (weighted)")
    ax.scatter(*STAGING_POINT, marker="*", s=220, c="gray", edgecolors="k", zorder=4, label="staging point")

    for pt in FAST_CHARGE_STATIONS:
        ax.scatter(*pt, marker="P", s=200, c="deepskyblue", edgecolors="k", zorder=4)
    for pt in SLOW_CHARGE_STATIONS:
        ax.scatter(*pt, marker="P", s=200, c="slategray", edgecolors="k", zorder=4)
    ax.scatter([], [], marker="P", c="deepskyblue", edgecolors="k", s=100, label="fast charger")
    ax.scatter([], [], marker="P", c="slategray", edgecolors="k", s=100, label="slow charger")

    trail_lines = [ax.plot([], [], "-", color=UE_COLORS[i], alpha=0.35, linewidth=1.5)[0]
                   for i in range(4)]
    ue_markers = [ax.scatter([0], [0], c=UE_COLORS[i], s=130, zorder=5,
                              edgecolors="k", linewidths=1.2) for i in range(4)]
    battery_texts = [ax.text(0, 0, "", fontsize=7, ha="center", va="bottom", zorder=6) for _ in range(4)]

    for i in range(4):
        ax.scatter([], [], c=UE_COLORS[i], s=60, label=f"UE{i}")
    ax.scatter([], [], marker="o", c="white", edgecolors="k", s=60, label="empty-handed")
    ax.scatter([], [], marker="s", c="white", edgecolors="k", s=60, label="carrying small")
    ax.scatter([], [], marker="D", c="white", edgecolors="k", s=60, label="carrying big")
    ax.legend(loc="lower left", fontsize=6.5, ncol=4)

    info_text = ax.text(0.015, 0.985, "", transform=ax.transAxes, va="top", ha="left",
                         fontsize=8.5, family="monospace",
                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.88))

    trails = [[] for _ in range(4)]

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        ue_pos = frame["ue_pos"]
        for i in range(4):
            trails[i].append(tuple(ue_pos[i]))
            if len(trails[i]) > TRAIL_LEN:
                trails[i].pop(0)
            xs, ys = zip(*trails[i])
            trail_lines[i].set_data(xs, ys)

            style = STATE_STYLE[frame["ue_box_type"][i]]
            halted = frame["ue_halted"][i]
            charging = frame["ue_state"][i] in ("to_charging", "charging")
            edge = "red" if halted else ("deepskyblue" if charging else "k")
            ue_markers[i].set_offsets([ue_pos[i]])
            ue_markers[i].set_paths([plt.matplotlib.markers.MarkerStyle(style["marker"]).get_path()])
            ue_markers[i].set_edgecolor(edge)
            ue_markers[i].set_linewidths(2.5 if (halted or charging) else 1.0)
            battery_texts[i].set_position((ue_pos[i][0], ue_pos[i][1] + 0.5))
            battery_texts[i].set_text(f"{frame['ue_battery'][i]:.0f}%")

        halted_str = "".join("H" if h else "." for h in frame["ue_halted"])
        state_str = " | ".join(
            f"UE{i}:{frame['ue_state'][i][:9]}({frame['ue_box_type'][i] or '-'},"
            f"v={frame['ue_speed'][i]:.2f})"
            for i in range(4)
        )
        info_text.set_text(
            f"tick {frame['tick']:4d} | mean SNR {frame['snr_per_ue'].mean():6.1f} dB | halted[{halted_str}] | "
            f"queue={frame['box_queue_len']:2d} | delivered={frame['deliveries_completed']:3d} | "
            f"charge trips={frame['charging_trips_started']:3d}\n{state_str}"
        )

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(buf)
    writer.close()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=1500)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", default="renders/ue_task_mobility_preview.mp4")
    args = p.parse_args()

    frames = rollout(args.ticks, args.seed)
    render_movie(frames, args.out, args.fps)
    deliveries = frames[-1]["deliveries_completed"]
    charges = frames[-1]["charging_trips_started"]
    print(f"Wrote {args.out} ({len(frames)} frames @ {args.fps}fps = {len(frames)/args.fps:.1f}s). "
          f"{deliveries} deliveries, {charges} charging trips over {len(frames)} ticks.")


if __name__ == "__main__":
    main()
