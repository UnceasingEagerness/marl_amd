import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from orbax.checkpoint import PyTreeCheckpointer, RestoreArgs
from rich.console import Console

# Import our custom modules
from env.jax_usv_env import JaxUSVEnv, EnvParams
from algorithms.flax_sac import Actor

console = Console()

def main():
    console.print("[bold cyan]RLSim V2: Multi-Agent Inference Engine[/bold cyan]")
    
    num_agents = 5
    seq_len = 10
    base_obs_dim = 72 + (num_agents - 1) * 5
    obs_dim = base_obs_dim * seq_len
    action_dim = 2
    
    layout = {
        "ego": {"start": 0, "dim": 8},
        "goal": {"start": 0, "dim": 8}, 
        "lidar": {"start": 8, "dim": 64},
        "auv_entities": {"start": 72, "dim": (num_agents - 1) * 5, "count": num_agents - 1, "feature_dim": 5},
        "moving_obstacles": {"start": 72, "dim": 0, "count": 0, "feature_dim": 5}
    }
    
    env = JaxUSVEnv()
    # Shorter distance, more time so they can reach it and we can watch them surround it!
    env_params = env.default_params.replace(num_agents=num_agents, map_size=800.0, num_obstacles=15, max_steps=1200)
    
    rng = jax.random.PRNGKey(42)
    jitted_map_gen = jax.jit(env.generate_map_bank, static_argnums=(1, 2, 3, 4))
    goals_bank, obstacles_bank = jitted_map_gen(
        rng, 
        int(env_params.num_agents), 
        int(env_params.num_obstacles), 
        float(env_params.map_size), 
        1
    )
    
    # Force all agents to share the exact same goal (Agent 0's goal)
    shared_goal = goals_bank[:, 0:1, :]
    goals_bank = jnp.repeat(shared_goal, num_agents, axis=1)
    
    env_params = env_params.replace(goals_bank=goals_bank, obstacles_bank=obstacles_bank)
    
    actor = Actor(layout=layout, action_dim=action_dim, action_scale=jnp.ones(action_dim), action_bias=jnp.zeros(action_dim))
    
    # ── Load Model ────────────────────────────────────────────────────────────
    ckpt_dir = os.path.abspath("/home/trizzz/AUV_Project/RLSim/multi_agent_nav_max/spread3/checkpoints_spread3/sac_actor_final")
    if not os.path.exists(ckpt_dir):
        console.print(f"[red]Error: Checkpoint {ckpt_dir} not found![/red]")
        return
        
    ckpt = PyTreeCheckpointer()
    
    dummy_obs = jnp.zeros((1, obs_dim))
    init_params = actor.init(jax.random.PRNGKey(0), dummy_obs)["params"]
    
    restore_args = jax.tree_util.tree_map(lambda _: RestoreArgs(restore_type=np.ndarray), init_params)
    raw_params = ckpt.restore(ckpt_dir, item=init_params, restore_args=restore_args)
    actor_params = jax.tree_util.tree_map(jnp.array, raw_params)
    
    console.print("[green]✔ Neural Network loaded.[/green]")
    
    # ── JIT Inference Functions ───────────────────────────────────────────────
    def get_action(params, obs):
        mean, _ = actor.apply({"params": params}, obs)
        return jnp.tanh(mean)
        
    jit_action = jax.jit(get_action)
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    
    # ── Run Simulation ────────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(42)
    reset_keys = jax.random.split(rng, 1)
    
    obs_batch, state_batch = vmap_reset(reset_keys, env_params)
    
    history_pos = []
    
    goals = state_batch.goal_pos[0]      # [N, 2]
    
    max_steps = 1200
    for step in range(max_steps):
        flat_obs = obs_batch.reshape(num_agents, obs_dim)
        actions = jit_action(actor_params, flat_obs)
        batched_actions = jnp.expand_dims(actions, 0)
        
        step_keys = jax.random.split(jax.random.PRNGKey(step), 1)
        obs_batch, state_batch, reward, done, _ = vmap_step(step_keys, state_batch, batched_actions, env_params)
        
        if step % 2 == 0:
            pos = state_batch.usv_state.eta[0, :, :2]
            history_pos.append(np.array(pos))
            
    # Print final analysis
    final_pos = history_pos[-1]
    dists = np.linalg.norm(final_pos - np.array(goals[0])[None, :], axis=1)
    angles = np.arctan2(final_pos[:, 1] - float(goals[0, 1]), final_pos[:, 0] - float(goals[0, 0]))
    sorted_angles = np.sort(angles)
    gaps = np.diff(sorted_angles)
    wrap_gap = sorted_angles[0] + 2.0 * np.pi - sorted_angles[-1]
    max_gap = np.max(np.append(gaps, wrap_gap))
    
    console.print(f"[bold cyan]Final Distances to Goal:[/bold cyan] {dists}")
    console.print(f"[bold cyan]Max Angular Gap:[/bold cyan] {max_gap:.2f} rad")
    
    # ── Rendering ─────────────────────────────────────────────────────────────
    console.print("[yellow]Rendering MP4/GIF Animation...[/yellow]")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('white')
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-1500, 1500)
    ax.grid(True, color='grey', linestyle='-', alpha=0.5)
    
    goal_arr = np.array(goals)
    ax.scatter(goal_arr[0, 0], goal_arr[0, 1], marker='*', color='green', s=400, label='Target (Stationary)')
    
    start_pos = history_pos[0]
    ax.scatter(start_pos[:, 0], start_pos[:, 1], marker='o', color='blue', s=100, label='Start')
        
    traj_lines = []
    boat_markers = []
    colors = ['red', 'orange', 'purple', 'magenta', 'brown']
    
    for i in range(num_agents):
        line, = ax.plot([], [], color=colors[i%len(colors)], linewidth=2)
        traj_lines.append(line)
        marker, = ax.plot([], [], 'o', color=colors[i%len(colors)], markersize=8)
        boat_markers.append(marker)
        
    def animate(frame):
        past_pos = np.array(history_pos[:frame+1])
        updated_artists = []
        for i in range(num_agents):
            traj_lines[i].set_data(past_pos[:, i, 0], past_pos[:, i, 1])
            boat_markers[i].set_data([past_pos[-1, i, 0]], [past_pos[-1, i, 1]])
            updated_artists.extend([traj_lines[i], boat_markers[i]])
        return updated_artists
        
    anim = animation.FuncAnimation(fig, animate, frames=len(history_pos), interval=50, blit=True)
    anim.save('/home/trizzz/.gemini/antigravity/brain/51b1597f-c9bf-41b0-a47d-c1293dbea528/artifacts/true_spread_eval.gif', writer='pillow', fps=30)
    console.print("[bold green]✔ Saved to artifacts[/bold green]")
    plt.close()

if __name__ == "__main__":
    main()
