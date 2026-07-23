import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from orbax.checkpoint import PyTreeCheckpointer, RestoreArgs
from rich.console import Console
from env.jax_usv_env import JaxUSVEnv
from algorithms.flax_sac import Actor

console = Console()

def run_simulation(num_agents_sim, out_filename):
    console.print(f"[bold cyan]Running Simulation for N={num_agents_sim}[/bold cyan]")
    
    # ── Config ────────────────────────────────────────────────────────
    max_agents_nn = 5
    seq_len = 10
    obs_dim_single = 92
    obs_dim_nn = obs_dim_single * seq_len
    action_dim = 2
    
    layout = {
        "ego": {"start": 0, "dim": 8},
        "goal": {"start": 0, "dim": 8}, 
        "lidar": {"start": 8, "dim": 64},
        "auv_entities": {"start": 72, "dim": (max_agents_nn - 1) * 5, "count": max_agents_nn - 1, "feature_dim": 5},
        "moving_obstacles": {"start": 72, "dim": 0, "count": 0, "feature_dim": 5}
    }
    
    env = JaxUSVEnv()
    env_params = env.default_params.replace(num_agents=num_agents_sim, map_size=500.0, num_obstacles=45)
    
    rng = jax.random.PRNGKey(111)
    jitted_map_gen = jax.jit(env.generate_map_bank, static_argnums=(1, 2, 3, 4))
    goals_bank, obstacles_bank, currents_bank = jitted_map_gen(rng, num_agents_sim, 45, 500.0, 1)
    
    num_dyn_obs = 6
    num_rl_agents = num_agents_sim - num_dyn_obs
    
    np.random.seed(111)
    map_size = 500
    half_size = 250
    spawn_center = np.array([-200, -200])
    goal_center = np.array([200, 200])
    placed_obstacles = []
    
    def is_valid_position(x, y, r):
        if abs(x) + r > half_size or abs(y) + r > half_size: return False
        if np.linalg.norm(np.array([x, y]) - spawn_center) < (r + 40.0): return False
        if np.linalg.norm(np.array([x, y]) - goal_center) < (r + 40.0): return False
        for ox, oy, orad in placed_obstacles:
            if np.linalg.norm(np.array([x, y]) - np.array([ox, oy])) < (orad + r + 5.0): return False
        return True

    static_obstacles = []
    grid_points = []
    step = map_size / 7
    for i in range(7):
        for j in range(7):
            grid_points.append((-half_size + step/2 + i*step, -half_size + step/2 + j*step))
    np.random.shuffle(grid_points)
    
    for base_x, base_y in grid_points:
        if len(static_obstacles) >= 45: break
        for _ in range(100):
            r = np.random.uniform(15.0, 35.0)
            x = base_x + np.random.uniform(-step/3, step/3)
            y = base_y + np.random.uniform(-step/3, step/3)
            if is_valid_position(x, y, r):
                static_obstacles.append((x, y, r))
                placed_obstacles.append((x, y, r))
                break
                
    obs_np = np.array(static_obstacles)
    huge_obstacles = jnp.array(obs_np)
    obstacles_bank = obstacles_bank.at[0].set(huge_obstacles)
    
    # tighter spaced spawns to fit 500x500 map
    starts_x_rl = jnp.array([-150, -100, -200, -150], dtype=jnp.float32)
    starts_y_rl = jnp.array([-100, -150, -150, -200], dtype=jnp.float32)
    angles_rl = jnp.array([jnp.pi/4, jnp.pi/4, jnp.pi/4, jnp.pi/4], dtype=jnp.float32)
    goals_x_rl = jnp.array([150, 100, 200, 150], dtype=jnp.float32)
    goals_y_rl = jnp.array([100, 150, 150, 200], dtype=jnp.float32)
    
    dyn_starts_x = []
    dyn_starts_y = []
    for _ in range(num_dyn_obs):
        while True:
            r = np.random.uniform(15.0, 40.0)
            x = np.random.uniform(-half_size + r, half_size - r)
            y = np.random.uniform(-half_size + r, half_size - r)
            if is_valid_position(x, y, r):
                dyn_starts_x.append(x)
                dyn_starts_y.append(y)
                placed_obstacles.append((x, y, r))
                break
    
    starts_x = jnp.concatenate([starts_x_rl, jnp.array(dyn_starts_x)])
    starts_y = jnp.concatenate([starts_y_rl, jnp.array(dyn_starts_y)])
    angles = jnp.concatenate([angles_rl, jax.random.uniform(rng, (num_dyn_obs,), minval=0, maxval=2*jnp.pi)])
    start_pos = jnp.stack([starts_x, starts_y, angles], axis=1)
    
    goals_x_dyn = jnp.zeros(num_dyn_obs)
    goals_y_dyn = jnp.zeros(num_dyn_obs)
    clustered_goals = jnp.stack([
        jnp.concatenate([goals_x_rl, goals_x_dyn]),
        jnp.concatenate([goals_y_rl, goals_y_dyn])
    ], axis=1)
    goals_bank = goals_bank.at[0].set(clustered_goals)
    
    env_params = env_params.replace(goals_bank=goals_bank, obstacles_bank=obstacles_bank, currents_bank=currents_bank)
    
    actor = Actor(layout=layout, action_dim=action_dim, action_scale=jnp.ones(action_dim), action_bias=jnp.zeros(action_dim))
    
    # ── Load Model ────────────────────────────────────────────────────────
    ckpt_dir = os.path.abspath("fresh/checkpoints_max_fresh/sac_actor_final")
    ckpt = PyTreeCheckpointer()
    dummy_obs = jnp.zeros((1, obs_dim_nn))
    init_params = actor.init(jax.random.PRNGKey(0), dummy_obs)["params"]
    restore_args = jax.tree_util.tree_map(lambda _: RestoreArgs(restore_type=np.ndarray), init_params)
    raw_params = ckpt.restore(ckpt_dir, item=init_params, restore_args=restore_args)
    actor_params = jax.tree_util.tree_map(jnp.array, raw_params)
    
    def get_action(params, obs):
        mean, _ = actor.apply({"params": params}, obs)
        return jnp.tanh(mean)
        
    jit_action = jax.jit(get_action)
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    
    reset_keys = jax.random.split(rng, 1)
    obs_batch, state_batch = vmap_reset(reset_keys, env_params)
    
    new_eta = start_pos[None, ...] # [1, N, 3]
    new_usv = state_batch.usv_state.replace(eta=new_eta)
    state_batch = state_batch.replace(usv_state=new_usv)
    
    dummy_actions = jnp.zeros((1, num_agents_sim, action_dim))
    step_keys = jax.random.split(jax.random.PRNGKey(999), 1)
    obs_batch, state_batch, _, _, _ = vmap_step(step_keys, state_batch, dummy_actions, env_params)
    
    history_pos = []
    obstacles = state_batch.obstacles[0]
    goals = state_batch.goal_pos[0]
    
    max_steps = 1500
    for step in range(max_steps):
        obs_env = obs_batch[0]
        obs_env_frames = obs_env.reshape(num_agents_sim, seq_len, -1)
        
        obs_nn_frames = jnp.zeros((num_agents_sim, seq_len, obs_dim_single))
        for i in range(num_agents_sim):
            recent_neighbors = obs_env_frames[i, -1, 72:].reshape(-1, 5)
            num_actual_neighbors = recent_neighbors.shape[0]
            
            dists = recent_neighbors[:, 1]**2 + recent_neighbors[:, 2]**2
            dists = jnp.where(recent_neighbors[:, 0] > 0.5, dists, 999999.0)
            
            top_k = min(4, num_actual_neighbors)
            top_indices = jnp.argsort(dists)[:top_k]
            
            obs_nn_frames = obs_nn_frames.at[i, :, :72].set(obs_env_frames[i, :, :72])
            for t in range(seq_len):
                frame_neighbors = obs_env_frames[i, t, 72:].reshape(-1, 5)
                top_neighbors = frame_neighbors[top_indices]
                
                padded_neighbors = jnp.zeros((4, 5))
                padded_neighbors = padded_neighbors.at[:top_k].set(top_neighbors)
                obs_nn_frames = obs_nn_frames.at[i, t, 72:92].set(padded_neighbors.flatten())
                
        obs_nn = obs_nn_frames.reshape(num_agents_sim, obs_dim_nn)
        actions = jit_action(actor_params, obs_nn)
        
        override_actions = jnp.stack([jnp.ones(num_dyn_obs)*0.8, jnp.zeros(num_dyn_obs)], axis=1)
        actions = actions.at[-num_dyn_obs:].set(override_actions)
        
        history_pos.append(np.array(state_batch.usv_state.eta[0, :, :2]))
        
        step_keys = jax.random.split(jax.random.PRNGKey(step), 1)
        obs_batch, state_batch, reward, done, _ = vmap_step(step_keys, state_batch, jnp.expand_dims(actions, 0), env_params)
        
    console.print("[yellow]Rendering Animation...[/yellow]")
    fig, ax_map = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('white')
    
    ax_map.set_facecolor('white')
    ax_map.set_xlim(-250, 250)
    ax_map.set_ylim(-250, 250)
    ax_map.set_title(f"Zoomed 4-Agent Swarm Navigation (Shared Goal)", color='black', fontsize=16)
    ax_map.tick_params(colors='black')
    ax_map.grid(True, color='#e0e0e0', alpha=0.8)
    
    obs_np = np.array(obstacles)
    for i in range(obs_np.shape[0]):
        rect = plt.Rectangle((obs_np[i, 0] - obs_np[i, 2], obs_np[i, 1] - obs_np[i, 2]), 2*obs_np[i, 2], 2*obs_np[i, 2], color='#bbbbbb', alpha=0.8)
        ax_map.add_patch(rect)
    
    goal_np = np.array(goals)
    for i in range(num_rl_agents):
        ax_map.scatter(goal_np[i, 0], goal_np[i, 1], color='green', s=300, marker='*')
        circle = plt.Circle((goal_np[i, 0], goal_np[i, 1]), 35.0, color='green', fill=False, linestyle='--')
        ax_map.add_patch(circle)
    
    colors = plt.cm.Set1(np.linspace(0, 1, num_rl_agents))
    
    lines_rl = [ax_map.plot([], [], color=c, lw=2, alpha=0.5)[0] for c in colors]
    points_rl = [ax_map.plot([], [], 'o', color=c, ms=16)[0] for c in colors]
    
    lines_dyn = [ax_map.plot([], [], color='gray', lw=2, alpha=0.5, linestyle='--')[0] for _ in range(num_dyn_obs)]
    points_dyn = [ax_map.plot([], [], 's', color='gray', ms=12)[0] for _ in range(num_dyn_obs)]
    
    Q_x, Q_y = np.meshgrid(np.linspace(-250, 250, 10), np.linspace(-250, 250, 10))
    Q_u = np.ones_like(Q_x) * 0.5
    Q_v = np.ones_like(Q_y) * 0.2
    ax_map.quiver(Q_x, Q_y, Q_u, Q_v, color='lightblue', alpha=0.4, scale=25, width=0.005, headwidth=4, headlength=6, headaxislength=5, label='Ocean Current')
    
    lines = lines_rl + lines_dyn
    points = points_rl + points_dyn
    
    def animate(frame):
        past_pos = np.array(history_pos[:frame+1])
        
        for i in range(num_agents_sim):
            lines[i].set_data(past_pos[:, i, 0], past_pos[:, i, 1])
            points[i].set_data([past_pos[-1, i, 0]], [past_pos[-1, i, 1]])
            
        return lines + points
    
    anim = animation.FuncAnimation(fig, animate, frames=np.arange(0, max_steps, 20), interval=100, blit=True)
    anim.save(out_filename, writer='pillow', fps=10)
    plt.close()
    console.print(f"[bold green]Saved {out_filename}[/bold green]")

if __name__ == "__main__":
    os.makedirs("visualizations_contributions", exist_ok=True)
    run_simulation(10, "visualizations_contributions/presentation_clean_zoomed.gif")
