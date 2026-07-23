with open('visualize_contribution2.py', 'r') as f:
    content = f.read()

content = content.replace("num_dyn_obs = 8", "num_dyn_obs = 2")
content = content.replace("jnp.ones(8)*0.8, jnp.zeros(8)", "jnp.ones(2)*0.8, jnp.zeros(2)")
content = content.replace("actions = actions.at[-8:].set(override_actions)", "actions = actions.at[-2:].set(override_actions)")
content = content.replace("num_agents_sim-8", "num_agents_sim-2")
content = content.replace("goal_np[:-8, 0]", "goal_np[:-2, 0]")
content = content.replace("goal_np[:-8, 1]", "goal_np[:-2, 1]")
content = content.replace("for _ in range(8)", "for _ in range(2)")
content = content.replace("for i in range(8)", "for i in range(2)")
content = content.replace("run_simulation(12,", "run_simulation(6,")
content = content.replace("contrib2_N4.mp4", "contrib2_N6.mp4")

with open('visualize_contribution2.py', 'w') as f:
    f.write(content)
