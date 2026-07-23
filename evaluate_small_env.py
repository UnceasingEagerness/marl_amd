"""
evaluate_small_env.py
Runs the trained MASAC policy in the compact 150m x 150m environment
and records a clean MP4 video.
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from orbax.checkpoint import PyTreeCheckpointer, RestoreArgs
from rich.console import Console
from env.jax_usv_env import JaxUSVEnv, EnvState
from env.jax_dynamics import USVState
from algorithms.flax_sac import Actor

console = Console()

# ── Hyper-parameters ──────────────────────────────────────────────────────────
MAX_AGENTS_NN  = 5          # Network was trained with 5 agents (DO NOT CHANGE)
NUM_RL_AGENTS  = 6         # We run 3 real RL agents
NUM_DYN_OBS    = 5          # 2 dynamic obstacles as env agents  (3+2=5 total)
NUM_AGENTS_SIM = NUM_RL_AGENTS + NUM_DYN_OBS   # = 5  matches trained network
SEQ_LEN        = 10
OBS_DIM_SINGLE = 92          # 8 ego + 64 lidar + (5-1)*5 neighbors = 92
ENV_OBS_DIM = 72 + (NUM_AGENTS_SIM - 1) * 5
OBS_DIM_NN     = OBS_DIM_SINGLE * SEQ_LEN
ACTION_DIM     = 2
MAP_SIZE       = 150.0      # 150m x 150m
HALF           = MAP_SIZE / 2.0
DIST_SCALE     = MAP_SIZE / 1000.0  # rescale dist obs so network sees training-range values
MAX_STEPS      = 2500
SEED           = 42
FPS            = 30
OUT_FILE       = "visualizations_contributions/small_env_navigation.mp4"

# ── Agent & obstacle colours ───────────────────────────────────────────────────
AGENT_COLORS = [
    '#e53935', # Red
    '#8e24aa', # Purple
    '#00897b', # Teal
    '#1e88e5', # Blue
    '#fdd835', # Yellow
    '#fb8c00', # Orange
]
DYN_COLOR    = '#ef5350'
STATIC_COLOR = '#607d8b'

# ── Network layout (matches training) ─────────────────────────────────────────
layout = {
    "ego":             {"start": 0,  "dim": 8},
    "goal":            {"start": 0,  "dim": 8},
    "lidar":           {"start": 8,  "dim": 64},
    "auv_entities":    {"start": 72, "dim": (MAX_AGENTS_NN - 1) * 5,
                        "count": MAX_AGENTS_NN - 1, "feature_dim": 5},
    "moving_obstacles":{"start": 72, "dim": 0, "count": 0, "feature_dim": 5},
}

# ── Static obstacle layout ────────────────────────────────────────────────────
# All obstacles clustered in the central band between spawn and goal
STATIC_OBS = [
    (-22,  12,  6),
    (  8, -18,  7),
    ( 22,  18,  6),
    (-18,  -8,  5),
    (  5,  28,  5),
    ( -5, -28,  6),
    ( 28, -10,  6),
]

# ── Dynamic obstacle start positions ──────────────────────────────────────
DYN_STARTS = [
    (-20,  10, np.pi/3),   # Obstacle 1
    ( 30, -15, np.pi),     # Obstacle 2
    (  0,  30, -np.pi/2),  # Obstacle 3 (Top middle, facing down)
    (-40,  40, -np.pi/4),  # Obstacle 4 (Top left, facing bottom-right)
    ( 40,  40, -np.pi),    # Obstacle 5 (Top right, facing left)
]

# (Optional) Update DYN_VELS to match 5 items, though your script overrides this later
DYN_VELS = [
    ( 1.2, -0.8),
    (-1.0,  1.0),
    ( 0.0, -1.0),
    ( 1.0, -1.0),
    (-1.0,  0.0),
]

# ── Spawn & goal positions for RL agents ──────────────────────────────────────
# ── Spawn & goal positions for RL agents ──────────────────────────────────────
RL_STARTS = [
    (-65, -45, np.pi/4),  # Agent 1
    (-55, -45, np.pi/4),  # Agent 2
    (-45, -45, np.pi/4),  # Agent 3
    (-65, -55, np.pi/4),  # Agent 4
    (-55, -55, np.pi/4),  # Agent 5
    (-45, -55, np.pi/4),  # Agent 6
]

RL_GOALS  = [
    ( 65,  55),           # Agent 1
    ( 55,  55),           # Agent 2
    ( 45,  55),           # Agent 3
    ( 65,  45),           # Agent 4
    ( 55,  45),           # Agent 5
    ( 45,  45),           # Agent 6
]

def build_env():
    env = JaxUSVEnv()
    env_params = env.default_params.replace(
        num_agents   = NUM_AGENTS_SIM,
        map_size     = MAP_SIZE,
        num_obstacles= len(STATIC_OBS),
        lidar_range  = 70.0,
        goal_radius  = 8.0,
        agent_collision_radius = 5.0,
        obs_collision_radius   = 10.0,   # larger = agents stay further from walls
        max_steps    = MAX_STEPS,
    )

    rng = jax.random.PRNGKey(SEED)
    jitted_map_gen = jax.jit(env.generate_map_bank, static_argnums=(1, 2, 3, 4))
    goals_bank, obstacles_bank, currents_bank = jitted_map_gen(
        rng, NUM_AGENTS_SIM, len(STATIC_OBS), MAP_SIZE, 1
    )

    # Override map bank slot 0 with our exact environment layout
    obs_np  = np.array([[ox, oy, r] for ox, oy, r in STATIC_OBS], dtype=np.float32)
    goals_np = np.zeros((NUM_AGENTS_SIM, 2), dtype=np.float32)
    for i, (gx, gy) in enumerate(RL_GOALS):
        goals_np[i] = [gx, gy]
    # Dynamic obstacle goals = origin (unused, they are overridden)
    for i in range(NUM_RL_AGENTS, NUM_AGENTS_SIM):
        goals_np[i] = [0.0, 0.0]

    obstacles_bank = obstacles_bank.at[0].set(jnp.array(obs_np))
    goals_bank     = goals_bank.at[0].set(jnp.array(goals_np))

    # Ocean current: (0.6, 0.4) m/s  →  matching preview arrows
    #curr = jnp.array([0.6, 0.4, 0.0], dtype=jnp.float32)   # shape [3]
    curr = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)   # shape [3]
    currents_bank = currents_bank.at[0].set(curr)

    env_params = env_params.replace(
        goals_bank    = goals_bank,
        obstacles_bank= obstacles_bank,
        currents_bank = currents_bank,
    )
    return env, env_params

def load_actor():
    actor = Actor(
        layout=layout,
        action_dim=ACTION_DIM,
        action_scale=jnp.ones(ACTION_DIM),
        action_bias=jnp.zeros(ACTION_DIM),
    )
    ckpt_dir = os.path.abspath("checkpoints_max/sac_actor_final")
    ckpt = PyTreeCheckpointer()
    dummy_obs   = jnp.zeros((1, OBS_DIM_NN))
    init_params = actor.init(jax.random.PRNGKey(0), dummy_obs)["params"]
    restore_args = jax.tree_util.tree_map(
        lambda _: RestoreArgs(restore_type=np.ndarray), init_params
    )
    raw_params   = ckpt.restore(ckpt_dir, item=init_params, restore_args=restore_args)
    actor_params = jax.tree_util.tree_map(jnp.array, raw_params)
    return actor, actor_params

def pad_obs_for_nn(obs_env_frames):
    """Pad neighbour slots to 4, and rescale distance so network sees training-range values."""
    obs_nn_frames = np.zeros((NUM_AGENTS_SIM, SEQ_LEN, OBS_DIM_SINGLE))
    obs_nn_frames[:, :, :72] = obs_env_frames[:, :, :72]

    # CRITICAL: rescale normalized distance (index 5 in ego block) so the
    # network receives a value in the training range (0.5-1.0) rather than
    # the tiny 150m-map value (~0.1).  Clip to 1.0 to stay in-distribution.
    obs_nn_frames[:, :, 5] = np.clip(
        obs_env_frames[:, :, 5] * DIST_SCALE, 0.0, 1.0
    )

    for i in range(NUM_AGENTS_SIM):
        for t in range(SEQ_LEN):
            neighbors = obs_env_frames[i, t, 72:].reshape(-1, 5)
            n_neighbors = neighbors.shape[0]
            top_k = min(4, n_neighbors)
            dists = neighbors[:, 1]**2 + neighbors[:, 2]**2
            dists = np.where(neighbors[:, 0] > 0.5, dists, 1e9)
            top_idx = np.argsort(dists)[:top_k]
            padded = np.zeros((4, 5))
            padded[:top_k] = neighbors[top_idx]
            obs_nn_frames[i, t, 72:92] = padded.flatten()

    return obs_nn_frames.reshape(NUM_AGENTS_SIM, OBS_DIM_NN)

def run_simulation():
    console.print("[bold cyan]Building environment...[/bold cyan]")
    env, env_params = build_env()

    console.print("[bold cyan]Loading actor checkpoint...[/bold cyan]")
    actor, actor_params = load_actor()

    @jax.jit
    def get_action(params, obs):
        mean, _ = actor.apply({"params": params}, obs)
        return jnp.tanh(mean)

    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step  = jax.vmap(env.step,  in_axes=(0, 0, 0, None))

    rng = jax.random.PRNGKey(SEED)
    reset_keys = jax.random.split(rng, 1)
    obs_batch, state_batch = vmap_reset(reset_keys, env_params)

    # Override initial positions for all agents
    all_starts = RL_STARTS + [(dx, dy, da) for dx, dy, da in DYN_STARTS]
    eta_np = np.array([[x, y, a] for x, y, a in all_starts], dtype=np.float32)
    new_eta  = jnp.array(eta_np)[None, ...]          # [1, N, 3]
    new_usv  = state_batch.usv_state.replace(eta=new_eta)
    state_batch = state_batch.replace(usv_state=new_usv)

    # Warm-up step
    dummy_actions = jnp.zeros((1, NUM_AGENTS_SIM, ACTION_DIM))
    step_keys = jax.random.split(jax.random.PRNGKey(999), 1)
    obs_batch, state_batch, _, _, _ = vmap_step(
        step_keys, state_batch, dummy_actions, env_params
    )

    # ── Rollout ───────────────────────────────────────────────────────────────
    history_pos   = []
    history_yaw   = []
    reached       = [False] * NUM_RL_AGENTS

    console.print(f"[bold cyan]Rolling out {MAX_STEPS} steps...[/bold cyan]")
    for step_i in range(MAX_STEPS):
        obs_env_frames = np.array(obs_batch[0]).reshape(NUM_AGENTS_SIM, SEQ_LEN, ENV_OBS_DIM)
        obs_nn = jnp.array(pad_obs_for_nn(obs_env_frames))

        actions = np.array(get_action(actor_params, obs_nn))

        # Override dynamic obstacles with simple straight drive + drift
        for d in range(NUM_DYN_OBS):
            idx = NUM_RL_AGENTS + d
            #actions[idx] = [0.6, np.sin(step_i * 0.03 + d * 2.1) * 0.4]
            actions[idx] = [0.15, 0.0]

            for i in range(NUM_RL_AGENTS):
                if reached[i]:
                    actions[i] = [0.0, 0.0]

        history_pos.append(np.array(state_batch.usv_state.eta[0, :, :2]))
        history_yaw.append(np.array(state_batch.usv_state.eta[0, :, 2]))

        step_keys  = jax.random.split(jax.random.PRNGKey(step_i + 1000), 1)
        obs_batch, state_batch, reward, done, info = vmap_step(
            step_keys, state_batch,
            jnp.expand_dims(jnp.array(actions), 0),
            env_params
        )

        # Check if RL agents reached goal
        for i in range(NUM_RL_AGENTS):
            if np.array(info["reached_goal"][0, i]):
                reached[i] = True

        if all(reached):
            console.print(f"[bold green]All agents reached goal at step {step_i}![/bold green]")
            # Record a few more frames so we can see the end
            for _ in range(60):
                history_pos.append(np.array(state_batch.usv_state.eta[0, :, :2]))
                history_yaw.append(np.array(state_batch.usv_state.eta[0, :, 2]))
            break

    console.print(f"[yellow]Recorded {len(history_pos)} frames. Rendering MP4...[/yellow]")

    # ── Rendering ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF, HALF)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('USV Swarm Navigation — 150m × 150m Environment', fontsize=13, fontweight='bold')
    ax.grid(True, color='#e0e0e0', linewidth=0.5)

    # Static obstacles
    for ox, oy, r in STATIC_OBS:
        c = plt.Circle((ox, oy), r, color=STATIC_COLOR, alpha=0.85, zorder=3)
        ax.add_patch(c)

    # Goal zones
    for i, (gx, gy) in enumerate(RL_GOALS):
        # Draw just the star at the goal location
        ax.plot(gx, gy, '*', color=AGENT_COLORS[i], markersize=14, zorder=5)

    

    # Spawn zone
    spawn_c = plt.Circle((-55, -55), 12, color='#1565c0',
                          fill=False, linestyle='--', linewidth=2, zorder=4)
    ax.add_patch(spawn_c)
    ax.text(-55, -55, 'Spawn', color='#1565c0', ha='center', va='center',
            fontsize=8, fontweight='bold')

    # Ocean current arrows (static background)
    xs = np.arange(-HALF + 12, HALF, 22)
    ys = np.arange(-HALF + 12, HALF, 22)
    XX, YY = np.meshgrid(xs, ys)
    ax.quiver(XX, YY, 0.6, 0.4, color='#90caf9', alpha=0.4,
              scale=18, width=0.003, headwidth=4, zorder=1)

    # ── Animated elements ─────────────────────────────────────────────────────
    # RL agent trails and dots
    trails_rl = [ax.plot([], [], '-', color=AGENT_COLORS[i], lw=1.5, alpha=0.5, zorder=6)[0]
                 for i in range(NUM_RL_AGENTS)]
    dots_rl   = [ax.plot([], [], 'o', color=AGENT_COLORS[i], ms=11, zorder=7,
                         markeredgecolor='white', markeredgewidth=1.5)[0]
                 for i in range(NUM_RL_AGENTS)]

    # Dynamic obstacle trails and squares
    trails_dyn = [ax.plot([], [], '--', color=DYN_COLOR, lw=1.2, alpha=0.4, zorder=6)[0]
                  for _ in range(NUM_DYN_OBS)]
    dots_dyn   = [ax.plot([], [], 's', color=DYN_COLOR, ms=9, zorder=7,
                          markeredgecolor='white', markeredgewidth=1.0)[0]
                  for _ in range(NUM_DYN_OBS)]

    # Step counter text
    step_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                        fontsize=10, color='#333333', va='top')

    # Legend
    for i in range(NUM_RL_AGENTS):
        ax.plot([], [], 'o', color=AGENT_COLORS[i], ms=9,
                label=f'AUV Agent {i+1}')
    ax.plot([], [], 's', color=DYN_COLOR, ms=8, label='Dynamic Obstacle')
    ax.plot([], [], color=STATIC_COLOR, marker='o', linestyle='', ms=9, label='Static Obstacle')
    ax.quiver([], [], [], [], color='#90caf9', label='Ocean Current')
    #ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    def animate(frame_idx):
        pos_frame = history_pos[frame_idx]   # [N, 2]

        trail_start = 0 #max(0, frame_idx - 80)
        trail_pos   = np.array(history_pos[trail_start:frame_idx + 1])

        for i in range(NUM_RL_AGENTS):
            if trail_pos.shape[0] > 1:
                trails_rl[i].set_data(trail_pos[:, i, 0], trail_pos[:, i, 1])
            dots_rl[i].set_data([pos_frame[i, 0]], [pos_frame[i, 1]])

        for d in range(NUM_DYN_OBS):
            idx = NUM_RL_AGENTS + d
            if trail_pos.shape[0] > 1:
                trails_dyn[d].set_data(trail_pos[:, idx, 0], trail_pos[:, idx, 1])
            dots_dyn[d].set_data([pos_frame[idx, 0]], [pos_frame[idx, 1]])

        step_text.set_text(f'Step: {frame_idx * 4}')
        return trails_rl + dots_rl + trails_dyn + dots_dyn + [step_text]

    total_frames = len(history_pos)
    frame_indices = np.arange(0, total_frames, 4)

    anim = animation.FuncAnimation(
        fig, animate,
        frames=frame_indices,
        interval=1000 // FPS,
        blit=True
    )

    os.makedirs("visualizations_contributions", exist_ok=True)
    writer = animation.FFMpegWriter(fps=FPS, bitrate=2000,
                                     extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    anim.save(OUT_FILE, writer=writer)
    plt.close()
    console.print(f"[bold green]✓ Saved MP4 → {OUT_FILE}[/bold green]")

if __name__ == "__main__":
    run_simulation()
