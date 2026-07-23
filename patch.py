with open('visualize_contribution1.py', 'r') as f:
    content = f.read()

content = content.replace(
    '    rng = jax.random.PRNGKey(42)\n    # Randomize Scenario (500m Zone)',
    '    rng = jax.random.PRNGKey(42)\n    jitted_map_gen = jax.jit(env.generate_map_bank, static_argnums=(1, 2, 3, 4))\n    goals_bank, obstacles_bank = jitted_map_gen(rng, num_agents_sim, 250, 1500.0, 1)\n    # Randomize Scenario (500m Zone)'
)

with open('visualize_contribution1.py', 'w') as f:
    f.write(content)
