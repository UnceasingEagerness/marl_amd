with open('visualize_contribution2.py', 'r') as f:
    content = f.read()

content = content.replace(
    'np.random.uniform(20.0, 70.0)',
    'np.random.uniform(15.0, 35.0)' # Slightly smaller to guarantee passage
)
content = content.replace(
    'ocean_current=jnp.array([1.2, 0.6, 0.0])',
    'ocean_current=jnp.array([0.5, 0.2, 0.0])'
)
content = content.replace(
    "Q_u = np.ones_like(Q_x) * 1.2\n    Q_v = np.ones_like(Q_y) * 0.6",
    "Q_u = np.ones_like(Q_x) * 0.5\n    Q_v = np.ones_like(Q_y) * 0.2"
)

with open('visualize_contribution2.py', 'w') as f:
    f.write(content)
