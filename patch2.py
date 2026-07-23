import re

with open('visualize_contribution2.py', 'r') as f:
    content = f.read()

# 1. Update EnvParams with ocean current
content = content.replace(
    'env_params = env.default_params.replace(num_agents=num_agents_sim, map_size=1500.0, num_obstacles=25)',
    'env_params = env.default_params.replace(num_agents=num_agents_sim, map_size=2000.0, num_obstacles=45, ocean_current=jnp.array([1.2, 0.6, 0.0]))'
)

# 2. Replace the obstacle generation block with the new 7x7 logic
old_block = """    rng = jax.random.PRNGKey(42)
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
    goals_bank = goals_bank.at[0].set(clustered_goals)"""

new_block = """    rng = jax.random.PRNGKey(111)
    jitted_map_gen = jax.jit(env.generate_map_bank, static_argnums=(1, 2, 3, 4))
    goals_bank, obstacles_bank = jitted_map_gen(rng, num_agents_sim, 45, 2000.0, 1)
    
    num_dyn_obs = 8
    num_rl_agents = num_agents_sim - num_dyn_obs
    
    import numpy as np
    np.random.seed(111)
    map_size = 2000
    half_size = 1000
    spawn_center = np.array([-800, -800])
    goal_center = np.array([800, 800])
    placed_obstacles = []
    
    def is_valid_position(x, y, r):
        if abs(x) + r > half_size or abs(y) + r > half_size: return False
        if np.linalg.norm(np.array([x, y]) - spawn_center) < (150 + r + 20.0): return False
        if np.linalg.norm(np.array([x, y]) - goal_center) < (150 + r + 20.0): return False
        for ox, oy, orad in placed_obstacles:
            if np.linalg.norm(np.array([x, y]) - np.array([ox, oy])) < (orad + r + 15.0): return False
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
            r = np.random.uniform(20.0, 70.0)
            x = base_x + np.random.uniform(-step/3, step/3)
            y = base_y + np.random.uniform(-step/3, step/3)
            if is_valid_position(x, y, r):
                static_obstacles.append((x, y, r))
                placed_obstacles.append((x, y, r))
                break
                
    obs_np = np.array(static_obstacles)
    huge_obstacles = jnp.array(obs_np)
    obstacles_bank = obstacles_bank.at[0].set(huge_obstacles)
    
    # RL Agents start at spawn, goals at goal
    starts_x_rl = jax.random.uniform(jax.random.split(rng)[0], (num_rl_agents,), minval=-900, maxval=-700)
    starts_y_rl = jax.random.uniform(jax.random.split(rng)[0], (num_rl_agents,), minval=-900, maxval=-700)
    angles_rl = jax.random.uniform(jax.random.split(rng)[0], (num_rl_agents,), minval=0, maxval=jnp.pi/2)
    
    goals_x_rl = jax.random.uniform(jax.random.split(rng)[0], (num_rl_agents,), minval=700, maxval=900)
    goals_y_rl = jax.random.uniform(jax.random.split(rng)[0], (num_rl_agents,), minval=700, maxval=900)
    
    # Dyn Obs start random
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
    
    # Dummy goals for dyn obs
    goals_x_dyn = jnp.zeros(num_dyn_obs)
    goals_y_dyn = jnp.zeros(num_dyn_obs)
    clustered_goals = jnp.stack([
        jnp.concatenate([goals_x_rl, goals_x_dyn]),
        jnp.concatenate([goals_y_rl, goals_y_dyn])
    ], axis=1)
    goals_bank = goals_bank.at[0].set(clustered_goals)"""

content = content.replace(old_block, new_block)

# 3. Action override for 8 dynamic obstacles
content = content.replace(
    'override_actions = jnp.array([[1.0, 0.0], [1.0, 0.0]]) # Max throttle, 0 steering\n        actions = actions.at[-2:].set(override_actions)',
    'override_actions = jnp.stack([jnp.ones(8)*0.8, jnp.zeros(8)], axis=1) # Moderate throttle, 0 steering\n        actions = actions.at[-8:].set(override_actions)'
)

# 4. Update plot code
content = content.replace(
    'ax_map.set_xlim(-350, 350)\n    ax_map.set_ylim(-350, 350)\n    ax_map.set_title(f"Randomized Swarm (N={num_agents_sim-2}) + Explicit Dynamic Obstacles", color=\'black\', fontsize=14, pad=15)',
    'ax_map.set_xlim(-1000, 1000)\n    ax_map.set_ylim(-1000, 1000)\n    ax_map.set_title(f"Swarm Navigation (N={num_agents_sim-8}) with Ocean Currents & Dynamic Obstacles", color=\'black\', fontsize=14, pad=15)'
)

content = content.replace(
    "ax_map.scatter(goal_np[:-2, 0], goal_np[:-2, 1], color='#32CD32', s=150, marker='*', label='Swarm Goals')",
    "ax_map.scatter(goal_np[:-8, 0], goal_np[:-8, 1], color='#32CD32', s=150, marker='*', label='Swarm Goals')"
)

content = content.replace(
    "colors = plt.cm.Dark2(np.linspace(0, 1, num_agents_sim-2))",
    "colors = plt.cm.Dark2(np.linspace(0, 1, num_agents_sim-8))"
)

content = content.replace(
    "lines_dyn = [ax_map.plot([], [], color='red', lw=2, alpha=0.7, linestyle='--')[0] for _ in range(2)]\n    points_dyn = [ax_map.plot([], [], '^', color='red', ms=12, markeredgecolor='black', label='Moving Obstacle' if i==0 else \"\")[0] for i in range(2)]",
    "lines_dyn = [ax_map.plot([], [], color='red', lw=2, alpha=0.5, linestyle='--')[0] for _ in range(8)]\n    points_dyn = [ax_map.plot([], [], '^', color='red', ms=12, markeredgecolor='black', label='Dynamic Obstacle' if i==0 else \"\")[0] for i in range(8)]\n    # Add Ocean Current Quiver\n    Q_x, Q_y = np.meshgrid(np.linspace(-900, 900, 10), np.linspace(-900, 900, 10))\n    Q_u = np.ones_like(Q_x) * 1.2\n    Q_v = np.ones_like(Q_y) * 0.6\n    ax_map.quiver(Q_x, Q_y, Q_u, Q_v, color='lightblue', alpha=0.4, scale=25, width=0.005, headwidth=4, headlength=6, headaxislength=5, label='Ocean Current')"
)

# Fix run_simulation calls
content = content.replace(
    'if __name__ == "__main__":\n    run_simulation(4, "visualizations_contributions/contrib1_N4.gif")\n    run_simulation(8, "visualizations_contributions/contrib1_N8.gif")',
    'if __name__ == "__main__":\n    run_simulation(12, "visualizations_contributions/contrib2_N4.mp4")'
)

# Change output format to mp4
content = content.replace(
    "anim.save(f'visualizations_contributions/contrib1_N{num_agents_sim}.gif', writer='imagemagick', fps=10)",
    "anim.save(out_filename, writer='ffmpeg', fps=10, bitrate=1800)"
)

# Change line thickness for RL agents
content = content.replace(
    "lines_rl = [ax_map.plot([], [], color=c, lw=2, alpha=0.7)[0] for c in colors]",
    "lines_rl = [ax_map.plot([], [], color=c, lw=3, alpha=1.0)[0] for c in colors]"
)

with open('visualize_contribution2.py', 'w') as f:
    f.write(content)
