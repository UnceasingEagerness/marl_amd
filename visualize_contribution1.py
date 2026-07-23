import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from orbax.checkpoint import PyTreeCheckpointer, RestoreArgs
from rich.console import Console
from env.jax_usv_env import JaxUSVEnv
from algorithms.flax_sac import Actor

console = Console()

def run_simulation(num_agents_sim, out_filename):
    console.print(f"[bold cyan]Running Simulation for N={num_agents_sim}[/bold cyan]")
    
    # ── Config ────────────────────────────────────────────────────────
    # The NN was trained on 5 agents (92 dims per frame). Sequence length = 10.
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
    # Map from -1500 to 1500
    env_params = env.default_params.replace(num_agents=num_agents_sim, map_size=1500.0, num_obstacles=25)
    
    rng = jax.random.PRNGKey(42)
    jitted_map_gen = jax.jit(env.generate_map_bank, static_argnums=(1, 2, 3, 4))
    goals_bank, obstacles_bank = jitted_map_gen(rng, num_agents_sim, 25, 1500.0, 1)
    # Randomize Scenario (500m Zone)
    
    # Generate massive varied static obstacles
    num_static = 25
    obs_x = jax.random.uniform(rng, (num_static,), minval=-300, maxval=300)
    obs_y = jax.random.uniform(rng, (num_static,), minval=-300, maxval=300)
    obs_r = jax.random.uniform(rng, (num_static,), minval=10.0, maxval=35.0)
    huge_obstacles = jnp.stack([obs_x, obs_y, obs_r], axis=1)
    obstacles_bank = obstacles_bank.at[0].set(huge_obstacles)
    
    # 2 explicit dynamic obstacles, the rest are RL swarm agents.
    num_dyn_obs = 2
    num_rl_agents = num_agents_sim - num_dyn_obs
    
    # Swarm starts random near bottom-left/center, goals random top-right
    starts_x_rl = jax.random.uniform(jax.random.split(rng)[0], (num_rl_agents,), minval=-250, maxval=0)
    starts_y_rl = jax.random.uniform(jax.random.split(rng)[0], (num_rl_agents,), minval=-250, maxval=0)
    angles_rl = jax.random.uniform(jax.random.split(rng)[0], (num_rl_agents,), minval=0, maxval=jnp.pi/2)
    
    # Dynamic obstacles spawn at edges and cross through
    starts_x_dyn = jnp.array([-250.0, 250.0])
    starts_y_dyn = jnp.array([100.0, -100.0])
    angles_dyn = jnp.array([0.0, jnp.pi]) # Moving right and left
    
    starts_x = jnp.concatenate([starts_x_rl, starts_x_dyn])
    starts_y = jnp.concatenate([starts_y_rl, starts_y_dyn])
    angles = jnp.concatenate([angles_rl, angles_dyn])
    
    start_pos = jnp.stack([starts_x, starts_y, angles], axis=1)
    
    # Swarm Goals
    goals_x_rl = jax.random.uniform(jax.random.split(rng)[0], (num_rl_agents,), minval=50, maxval=300)
    goals_y_rl = jax.random.uniform(jax.random.split(rng)[0], (num_rl_agents,), minval=50, maxval=300)
    goals_x_dyn = jnp.array([300.0, -300.0]) # Goals for dynamic obs so they drive straight
    goals_y_dyn = jnp.array([100.0, -100.0])
    
    clustered_goals = jnp.stack([
        jnp.concatenate([goals_x_rl, goals_x_dyn]),
        jnp.concatenate([goals_y_rl, goals_y_dyn])
    ], axis=1)
    goals_bank = goals_bank.at[0].set(clustered_goals)
    
    env_params = env_params.replace(goals_bank=goals_bank, obstacles_bank=obstacles_bank)
    
    actor = Actor(layout=layout, action_dim=action_dim, action_scale=jnp.ones(action_dim), action_bias=jnp.zeros(action_dim))
    
    # ── Load Model ────────────────────────────────────────────────────────
    ckpt_dir = os.path.abspath("checkpoints_max/sac_actor_final")
    ckpt = PyTreeCheckpointer()
    dummy_obs = jnp.zeros((1, obs_dim_nn))
    init_params = actor.init(jax.random.PRNGKey(0), dummy_obs)["params"]
    restore_args = jax.tree_util.tree_map(lambda _: RestoreArgs(restore_type=np.ndarray), init_params)
    raw_params = ckpt.restore(ckpt_dir, item=init_params, restore_args=restore_args)
    actor_params = jax.tree_util.tree_map(jnp.array, raw_params)
    
    # ── JIT Functions ─────────────────────────────────────────────────────
    def get_action(params, obs):
        mean, _ = actor.apply({"params": params}, obs)
        return jnp.tanh(mean)
        
    def get_mean_action_single(params, obs):
        mean, _ = actor.apply({"params": params}, jnp.expand_dims(obs, 0))
        return jnp.tanh(mean[0])
        
    jit_action = jax.jit(get_action)
    compute_gradients = jax.jacobian(get_mean_action_single, argnums=1)
    
    steps_alpha = 10
    alphas = jnp.linspace(0.0, 1.0, steps_alpha)
    
    def compute_integrated_gradients(params, obs):
        baseline = jnp.zeros_like(obs)
        diff = obs - baseline
        def step_fn(alpha):
            return compute_gradients(params, baseline + alpha * diff)
        grads = jax.vmap(step_fn)(alphas)
        return jnp.mean(grads, axis=0) * diff
        
    compute_saliency = jax.jit(compute_integrated_gradients)
    
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    
    # ── Simulate ──────────────────────────────────────────────────────────
    reset_keys = jax.random.split(rng, 1)
    obs_batch, state_batch = vmap_reset(reset_keys, env_params)
    
    # Force Start Positions
    new_eta = start_pos[None, ...] # [1, N, 3]
    new_usv = state_batch.usv_state.replace(eta=new_eta)
    state_batch = state_batch.replace(usv_state=new_usv)
    
    # Dummy step to recalculate obs_batch based on forced positions
    dummy_actions = jnp.zeros((1, num_agents_sim, action_dim))
    step_keys = jax.random.split(jax.random.PRNGKey(999), 1)
    obs_batch, state_batch, _, _, _ = vmap_step(step_keys, state_batch, dummy_actions, env_params)
    
    history_pos = []
    history_lidar = []
    history_sal_lidar = []
    history_sal_agents = []
    
    obstacles = state_batch.obstacles[0]
    goals = state_batch.goal_pos[0]
    
    max_steps = 300
    for step in range(max_steps):
        obs_env = obs_batch[0] # [num_agents_sim, (72 + (N-1)*5) * 10]
        obs_env_frames = obs_env.reshape(num_agents_sim, seq_len, -1) # [N, 10, env_features]
        
        # Pad/Truncate to 5 agents for NN (closest 4 neighbors)
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
                
                # Pad to exactly 4 neighbors (20 elements) if we have fewer
                padded_neighbors = jnp.zeros((4, 5))
                padded_neighbors = padded_neighbors.at[:top_k].set(top_neighbors)
                
                obs_nn_frames = obs_nn_frames.at[i, t, 72:92].set(padded_neighbors.flatten())
                
        obs_nn = obs_nn_frames.reshape(num_agents_sim, obs_dim_nn)
        actions = jit_action(actor_params, obs_nn)
        
        # Override last 2 agents to act as explicit linear dynamic obstacles
        override_actions = jnp.array([[1.0, 0.0], [1.0, 0.0]]) # Max throttle, 0 steering
        actions = actions.at[-2:].set(override_actions)
        
        # XAI for Ego Agent 0
        ego_obs = obs_nn[0]
        jacobian = compute_saliency(actor_params, ego_obs) # [2, 920]
        sal_mag = jnp.sum(jnp.abs(jacobian), axis=0) # [920]
        
        # We take the saliency over the most recent frame (last 92 elements)
        sal_mag_recent = sal_mag[-92:]
        
        sal_lidar = sal_mag_recent[8:72]
        sal_agents = sal_mag_recent[72:92].reshape(4, 5).sum(axis=1)
        
        history_pos.append(np.array(state_batch.usv_state.eta[0, :, :2]))
        history_lidar.append(np.array(obs_nn_frames[0, -1, 8:72]))
        history_sal_lidar.append(np.array(sal_lidar))
        history_sal_agents.append(np.array(sal_agents))
        
        step_keys = jax.random.split(jax.random.PRNGKey(step), 1)
        obs_batch, state_batch, reward, done, _ = vmap_step(step_keys, state_batch, jnp.expand_dims(actions, 0), env_params)
        
    # ── Render ────────────────────────────────────────────────────────────
    console.print("[yellow]Rendering Animation...[/yellow]")
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.2, 1])
    
    # Panel 1: Global Map
    ax_map = fig.add_subplot(gs[0:, 0])
    ax_map.set_facecolor('white')
    ax_map.set_xlim(-350, 350)
    ax_map.set_ylim(-350, 350)
    ax_map.set_title(f"Randomized Swarm (N={num_agents_sim-2}) + Explicit Dynamic Obstacles", color='black', fontsize=14, pad=15)
    ax_map.tick_params(colors='black')
    ax_map.grid(True, color='#e0e0e0', alpha=0.8)
    
    obs_np = np.array(obstacles)
    for i in range(obs_np.shape[0]):
        circ = plt.Circle((obs_np[i, 0], obs_np[i, 1]), obs_np[i, 2], color='#555555', alpha=0.6)
        ax_map.add_patch(circ)
    
    # Add dummy patch for legend
    ax_map.scatter([], [], color='#555555', s=100, alpha=0.6, label='Static Obstacles')
        
    goal_np = np.array(goals)
    ax_map.scatter(goal_np[:-2, 0], goal_np[:-2, 1], color='#32CD32', s=150, marker='*', label='Swarm Goals')
    
    colors = plt.cm.Dark2(np.linspace(0, 1, num_agents_sim-2))
    
    # RL Agents
    lines_rl = [ax_map.plot([], [], color=c, lw=2, alpha=0.7)[0] for c in colors]
    points_rl = [ax_map.plot([], [], 'o', color=c, ms=8, markeredgecolor='black')[0] for c in colors]
    points_rl[0].set_markeredgecolor('red')
    points_rl[0].set_markeredgewidth(2)
    points_rl[0].set_markersize(12)
    
    # Explicit Dynamic Obstacles
    lines_dyn = [ax_map.plot([], [], color='red', lw=2, alpha=0.7, linestyle='--')[0] for _ in range(2)]
    points_dyn = [ax_map.plot([], [], '^', color='red', ms=12, markeredgecolor='black', label='Moving Obstacle' if i==0 else "")[0] for i in range(2)]
    
    lines = lines_rl + lines_dyn
    points = points_rl + points_dyn
    ax_map.legend(loc='upper left', fontsize=10)
    
    # Panel 2: STAE Attention
    ax_stae = fig.add_subplot(gs[0, 1])
    ax_stae.set_facecolor('white')
    ax_stae.set_title("Ego Attention to Nearest 4 Neighbors (Saliency Matrix)", color='black', fontsize=12)
    ax_stae.tick_params(colors='black')
    im_stae = ax_stae.imshow(np.zeros((1, 4)), cmap='Reds', vmin=0, vmax=0.1, aspect='auto')
    ax_stae.set_yticks([])
    ax_stae.set_xticks(np.arange(4))
    ax_stae.set_xticklabels([f"Neighbor {i+1}" for i in range(4)])
    
    # Panel 3: LiDAR + Saliency (Polar)
    ax_lidar = fig.add_subplot(gs[1, 1], polar=True)
    ax_lidar.set_facecolor('white')
    ax_lidar.set_title("Ego LiDAR Occupancy & GradCAM", color='black', fontsize=12, pad=15)
    ax_lidar.tick_params(colors='black')
    ax_lidar.set_ylim(0, 1)
    ax_lidar.set_theta_zero_location("N")
    ax_lidar.set_theta_direction(-1)
    
    angles = np.linspace(-np.pi, jnp.pi, 64, endpoint=False)
    bars_lidar = ax_lidar.bar(angles, np.ones(64), width=2*np.pi/64, color='#b0e0e6', alpha=0.6)
    
    def animate(frame):
        past_pos = np.array(history_pos[:frame+1])
        
        for i in range(num_agents_sim):
            lines[i].set_data(past_pos[:, i, 0], past_pos[:, i, 1])
            points[i].set_data([past_pos[-1, i, 0]], [past_pos[-1, i, 1]])
            
        sal_a = history_sal_agents[frame]
        im_stae.set_data(sal_a.reshape(1, -1))
        im_stae.set_clim(vmin=0, vmax=max(0.01, np.max(sal_a)*1.2))
        
        lidar_ranges = history_lidar[frame]
        sal_l = history_sal_lidar[frame]
        
        max_sal = max(0.01, np.max(sal_l))
        for j, bar in enumerate(bars_lidar):
            bar.set_height(lidar_ranges[j])
            intensity = sal_l[j] / max_sal
            r = min(1.0, 0.5 + intensity*0.5)
            g = max(0.0, 0.8 - intensity*0.8)
            b = max(0.0, 0.9 - intensity*0.9)
            bar.set_color((r, g, b, 0.8))
            
        return lines + points + [im_stae] + list(bars_lidar)
    
    # Save a static preview image of the environment
    animate(0)
    plt.savefig(f'visualizations_contributions/contrib1_N{num_agents_sim}_preview.png', dpi=300, bbox_inches='tight')
    console.print(f"[green]Saved visualizations_contributions/contrib1_N{num_agents_sim}_preview.png[/green]")
    
    anim = animation.FuncAnimation(fig, animate, frames=max_steps, interval=100, blit=True)
    anim.save(f'visualizations_contributions/contrib1_N{num_agents_sim}.gif', writer='imagemagick', fps=10)
    plt.close()
    console.print(f"[bold green]Saved {out_filename}[/bold green]")

if __name__ == "__main__":
    os.makedirs("visualizations_contributions", exist_ok=True)
    run_simulation(4, "visualizations_contributions/contrib1_N4.gif")
    run_simulation(8, "visualizations_contributions/contrib1_N8.gif")
